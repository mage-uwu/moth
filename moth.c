// moth.c: Moth, a tiny ternary Monarch Mixer char-level LM in one file of pure C (nanoGPT style).
//
//   block(x) (the M2 layer of Monarch Mixer, Fu et al. 2023, causal):
//     x += M2( K * M1(rms x) )              sequence mixing: M1, M2 causal T x T Monarchs, K a [T][D] gain
//     x += Mon_d( gelu(Mon_u(rms x)) )      dimension mixing: D x D Monarchs
//   Mon: view a width-d vector as an MxM tile (d = M^2). R mixes along each row, L along each column.
//        That is P L P^T R. The permutations are never materialised: a column op on a row-major tile is
//        lane-wise vector math with weights stored [o][i][lane], so every loop streams contiguous int8.
//   The sequence Monarchs are block Monarchs over time (T = chunks x offsets): a lower-triangular block
//   diagonal for within-chunk mixing plus P L P^T R with strictly lower L for earlier chunks, which is
//   causal and reaches every past position. All Monarch weights are ternary {-1,0,1} with STE.
//
// forward : ternary {-1,0,1} Monarch weights (absmean) x int8 activations (per-token absmax), STE. A
//           ternary dot over 16 int8 terms fits in int16, so those accumulators use 32 lanes, not 16.
//           Absmax codes are scale-invariant, so RMSNorm, R->L requantisation and the gate all fold into
//           per-token scales. No float tensor is written that the backward doesn't need. Sequence mixing
//           runs on fp32 rows: a per-token int8 scale can't be shared by a sum over tokens.
// backward: gradients are int8 with a delayed per-tensor scale (last step's amax, as in FP8 training) and
//           stochastic rounding. There is no amax pre-pass and no barrier inside a Monarch. The input's per-token
//           scale is folded into the gradient, so dW = g8^T x8 reuses the forward's int8 codes.
//           dx = W^T g8 (ternary x int8, int16) and dW (int8 x int8) accumulate in per-thread int32 slabs.
//
// cc -O3 -march=native -fopenmp moth.c -o moth -lm && ./moth input.txt
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#define TID omp_get_thread_num()
#else
#define TID 0
#define omp_get_max_threads() 1
#endif

#ifndef M
#define M 16            // monarch block size
#endif
#define D (M * M)       // model width
#define W3 (M * M * M)  // weights per monarch factor
#ifndef L
#define L 4             // layers
#endif
#ifndef T
#define T 128           // context = long-conv length (power of 2)
#endif
#define B 16            // batch size
#define N (B * T)       // tokens per batch
#ifndef STEPS
#define STEPS 3000
#endif
#define LR 3e-3f

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 40) / 16777216.f; }
static float randn(void) { return sqrtf(-2 * logf(urand() + 1e-9f)) * cosf(6.2831853f * urand()); }
static uint32_t hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
static float *fa(size_t n) { return calloc(n, sizeof(float)); }
static int8_t *ia(size_t n) { return calloc(n, 1); }
static int NT;          // threads
static uint32_t seed;   // per-step stochastic rounding seed

typedef struct { float *w, *g, *m, *v; int n; } P;   // master weight, grad, adam moments
static P ps[1 + 11 * L]; static int np;   // embedding + 11 tensors per layer
static P *param(int n, float sd) {
    P *p = &ps[np++]; p->n = n; p->w = fa(n); p->g = fa(n); p->m = fa(n); p->v = fa(n);
    for (int i = 0; i < n; i++) p->w[i] = sd * randn();
    return p;
}

// ---- quantizers ------------------------------------------------------------------------------
static float tern(const float *w, int8_t *q, int n, int st) {        // absmean ternary, returns scale
    float s = 1e-8f; for (int i = 0; i < n; i++) s += fabsf(w[i * st]); s /= n;
    for (int i = 0; i < n; i++) { float r = roundf(w[i * st] / s); q[i * st] = r > 1 ? 1 : r < -1 ? -1 : r; }
    return s;
}
static float q8t(const float *x, int8_t *q, float mul) {              // per-token absmax int8; scale*mul
    float m = 1e-8f; for (int i = 0; i < D; i++) m = fmaxf(m, fabsf(x[i]));
    float is = 127 / m; for (int i = 0; i < D; i++) q[i] = (int8_t)rintf(x[i] * is);
    return m * mul / 127;
}
static int8_t sr8(float v, uint32_t idx) {                            // stochastic round + saturate
    float r = floorf(v + (hash(idx ^ seed) >> 8) * 0x1p-24f);
    return r > 127 ? 127 : r < -127 ? -127 : (int8_t)r;
}
typedef struct { float prev, cur[64]; } Amax;                          // delayed gradient scale
static float gscale(Amax *a) { return (a->prev > 0 ? a->prev : 1e-3f) / 127; }
static void roll(Amax *a) { a->prev = 0; for (int t = 0; t < NT; t++) { a->prev = fmaxf(a->prev, a->cur[t]); a->cur[t] = 0; } }

// ---- Monarch: R (row blocks, weights [b][i][o]) then L (column blocks, weights [o][i][lane]) ----
typedef struct { P *r, *l; int8_t rq[W3], rt[W3], lq[W3], *mq; float rs, ls, *ms; Amax gl, gr; int32_t *dw; uint32_t salt; } Mon;
static void mon_init(Mon *m, float sd, uint32_t salt) {
    m->r = param(W3, sd / sqrtf(M)); m->l = param(W3, 1 / sqrtf(M));
    m->mq = ia(N * D); m->ms = fa(N); m->dw = calloc((size_t)NT * 2 * W3, 4); m->salt = salt * 0x9E3779B9u;
}
static void mon_prep(Mon *m) {                                              // once per step, O(params):
    m->rs = tern(m->r->w, m->rq, W3, 1); m->ls = tern(m->l->w, m->lq, W3, 1);  // a 4KB ternary transpose so R^T
    for (int b = 0; b < M; b++) for (int i = 0; i < M; i++) for (int o = 0; o < M; o++)  // is broadcast-accumulate too
        m->rt[(b * M + o) * M + i] = m->rq[(b * M + i) * M + o];
}
// token t: x = int8 codes with scale sx -> y (float). Stores the int8 mid codes for the backward.
static void mon_fwd(Mon *m, const int8_t *x, float sx, float *y, int t) {
    int16_t z[D] = {0}, acc[D] = {0}; int mx = 1; int8_t *q = m->mq + t * D;  // |ternary.int8| <= 16*127: int16
    for (int b = 0; b < M; b++) for (int i = 0; i < M; i++) {                 // R: z_b += x_bi * R_bi.
        int xi = x[b * M + i]; const int8_t *w = m->rq + (b * M + i) * M; int16_t *zb = z + b * M;
        for (int o = 0; o < M; o++) zb[o] += xi * w[o];
    }
    for (int k = 0; k < D; k++) mx = abs(z[k]) > mx ? abs(z[k]) : mx;        // requantise int16 -> int8 directly
    float qs = 127.f / mx, sm = m->ms[t] = sx * m->rs * mx / 127;
    for (int k = 0; k < D; k++) q[k] = (int8_t)rintf(z[k] * qs);
    for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) {                 // L: row o += w_oi (.) row i, lane-wise
        const int8_t *w = m->lq + (o * M + i) * M, *qi = q + i * M; int16_t *a = acc + o * M;
        for (int b = 0; b < M; b++) a[b] += w[b] * qi[b];
    }
    for (int k = 0; k < D; k++) y[k] = acc[k] * sm * m->ls;
}
// token t: dy (float) -> dx (float); int8 x int8 dW into thread tid's int32 slab
static void mon_bwd(Mon *m, const int8_t *x, float sx, const float *dy, float *dx, int t, int tid) {
    const int8_t *q = m->mq + t * D; float sm = m->ms[t], gl = gscale(&m->gl), gr = gscale(&m->gr), am = 0;
    int8_t g[D]; int16_t dm[D] = {0}; int32_t *dwr = m->dw + (size_t)tid * 2 * W3, *dwl = dwr + W3;
    for (int k = 0; k < D; k++) { float v = dy[k] * sm; am = fmaxf(am, fabsf(v)); g[k] = sr8(v / gl, (t * D + k) ^ m->salt); }
    m->gl.cur[tid] = fmaxf(m->gl.cur[tid], am);
    for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) {                 // L^T g and dL, lane-wise
        const int8_t *w = m->lq + (o * M + i) * M, *go = g + o * M, *qi = q + i * M;
        int16_t *a = dm + i * M; int32_t *dl = dwl + (o * M + i) * M;
        for (int b = 0; b < M; b++) { a[b] += w[b] * go[b]; dl[b] += go[b] * qi[b]; }
    }
    float c = gl * m->ls / sm * sx / gr; am = 0;                             // fold sx so dR reuses x's codes
    for (int k = 0; k < D; k++) { float v = dm[k] * c; am = fmaxf(am, fabsf(v)); g[k] = sr8(v, (t * D + k) ^ ~m->salt); }
    m->gr.cur[tid] = fmaxf(m->gr.cur[tid], am * gr);
    float cx = gr * m->rs / sx; int16_t ax[D] = {0};
    for (int b = 0; b < M; b++) for (int o = 0; o < M; o++) {                 // R^T g and dR, broadcast g_bo
        int go = g[b * M + o]; const int8_t *w = m->rt + (b * M + o) * M, *xb = x + b * M;
        int16_t *a = ax + b * M; int32_t *dr = dwr + (b * M + o) * M;           // dR slab is [b][o][i]
        for (int i = 0; i < M; i++) { a[i] += go * w[i]; dr[i] += xb[i] * go; }
    }
    for (int k = 0; k < D; k++) dx[k] = ax[k] * cx;
}
static void mon_reduce(Mon *m) {                                             // sum thread slabs, O(params)
    float gr = gscale(&m->gr), gl = gscale(&m->gl);
    #pragma omp parallel for
    for (int k = 0; k < W3; k++) {
        int32_t sr = 0, sl = 0;
        for (int t = 0; t < NT; t++) { int32_t *s = m->dw + (size_t)t * 2 * W3; sr += s[k]; sl += s[W3 + k]; s[k] = s[W3 + k] = 0; }
        int b = k / (M * M), o = k / M % M, i = k % M;
        m->r->g[(b * M + i) * M + o] += sr * gr; m->l->g[k] += sl * gl;
    }
    roll(&m->gr); roll(&m->gl);
}

// ---- RMSNorm (no gain). Forward is folded into q8t's scale; backward recomputes y = x r ---------
static float rinv(const float *x) { float s = 0; for (int i = 0; i < D; i++) s += x[i] * x[i]; return 1 / sqrtf(s / D + 1e-5f); }
static void rms_bwd(const float *x, float r, const float *dy, float *dx) {   // dx += d rms(x)
    float dot = 0; for (int i = 0; i < D; i++) dot += dy[i] * x[i]; dot *= r * r / D;
    for (int i = 0; i < D; i++) dx[i] += (dy[i] - x[i] * dot) * r;
}

// ---- causal sequence Monarch: T x T, shared by the channels of a head ---------------------------
// Position t = b TS + i: chunk b of NB, offset i of TS. A single Monarch P L P^T R can't be causal and
// complete: its (c, o) <- (b, i) entry is L_o[c][b] R_b[o][i], and R_b would have to be triangular for
// its own chunk yet dense for later ones. So the causal Monarch is the sum of two:
//   y[c][o] = sum_{i <= o} A_c[o][i] x[c][i]                    within the chunk (A lower-triangular)
//           + sum_{b < c} L_o[c][b] sum_i R_b[o][i] x[b][i]     earlier chunks   (R dense, L strictly lower)
// which reaches every (b, i) <= (c, o). Loops run over rows of D channels, so the P's stay implicit.
// Ternary absmean weights with STE, times a fixed 1 / sqrt(fan-in) per row. The signal is fp32 (per-token
// int8 scales don't survive mixing across tokens), so the weights reduce to signed adds of D-wide rows.
static int TS, NB;
#ifndef H
#define H 1                                   // heads: channel groups with their own M1, M2 (M2 shares one)
#endif
#define HD (D / H)
#define CW (HD < 64 ? HD : 64)                // channels per task
static int SA, SL;                            // weights per head and factor: A, R are [NB][TS][TS], L is [TS][NB][NB]
typedef struct { P *w[3]; float *e[3], *dw; } SMon;       // A, R, L: latent, effective, per-thread grad slabs
static int sm_bs(int f) { return f == 2 ? NB : TS; }
static int sm_n(int f) { return f == 2 ? SL : SA; }
static int sm_fan(int f, int row, int col) {              // fan-in of a row, or 0 if (row, col) is masked
    if (f == 0) return col <= row ? row + 1 : 0;
    if (f == 1) return TS;
    return col < row ? row : 0;
}
#define SW (H * (2 * SA + SL))                             // grad slab size
static float *sm_slab(float *dw, int f, int h) { return dw + (f == 0 ? 0 : f == 1 ? H * SA : 2 * H * SA) + h * sm_n(f); }
static void smon_init(SMon *m) {
    for (int f = 0; f < 3; f++) { m->w[f] = param(H * sm_n(f), 1 / sqrtf(sm_bs(f))); m->e[f] = fa(H * sm_n(f)); }
    m->dw = fa((size_t)NT * SW);
}
static void smon_prep(SMon *m) {                           // ternarise the unmasked entries, per head and factor
    for (int f = 0; f < 3; f++) for (int h = 0; h < H; h++) {
        int n = sm_n(f), bs = sm_bs(f); const float *w = m->w[f]->w + h * n; float *q = m->e[f] + h * n, s = 1e-8f, cnt = 1e-8f;
        for (int k = 0; k < n; k++) if (sm_fan(f, k / bs % bs, k % bs)) s += fabsf(w[k]), cnt++;
        s /= cnt;
        for (int k = 0; k < n; k++) {
            int fan = sm_fan(f, k / bs % bs, k % bs); float r = roundf(w[k] / s);
            q[k] = fan ? (r > 1 ? 1 : r < -1 ? -1 : r) * s / sqrtf(fan) : 0;
        }
    }
}
// one sequence (rows of stride D), channels [c0, c1) of head h: x -> y; mid = R x (kept for backward)
static void smon_fwd(const SMon *m, const float *x, float *mid, float *y, int h, int c0, int c1) {
    const float *A = m->e[0] + h * SA, *R = m->e[1] + h * SA, *Lw = m->e[2] + h * SL;
    for (int b = 0; b < NB; b++) for (int o = 0; o < TS; o++) {
        float *u = mid + (b * TS + o) * D, *z = y + (b * TS + o) * D;
        for (int ch = c0; ch < c1; ch++) u[ch] = z[ch] = 0;
        for (int i = 0; i < TS; i++) {
            float wr = R[(b * TS + o) * TS + i], wa = A[(b * TS + o) * TS + i]; const float *v = x + (b * TS + i) * D;
            for (int ch = c0; ch < c1; ch++) { u[ch] += wr * v[ch]; z[ch] += wa * v[ch]; }
        }
    }
    for (int c = 1; c < NB; c++) for (int o = 0; o < TS; o++) {
        float *z = y + (c * TS + o) * D;
        for (int b = 0; b < c; b++) {
            float w = Lw[(o * NB + c) * NB + b]; const float *v = mid + (b * TS + o) * D;
            for (int ch = c0; ch < c1; ch++) z[ch] += w * v[ch];
        }
    }
}
// dy -> dx (dmid is scratch), weight grads into slab dw
static void smon_bwd(const SMon *m, const float *x, const float *mid, const float *dy, float *dmid, float *dx, int h, int c0, int c1, float *dw) {
    const float *A = m->e[0] + h * SA, *R = m->e[1] + h * SA, *Lw = m->e[2] + h * SL;
    float *dA = sm_slab(dw, 0, h), *dR = sm_slab(dw, 1, h), *dL = sm_slab(dw, 2, h);
    for (int b = 0; b < NB; b++) for (int o = 0; o < TS; o++) {
        float *u = dmid + (b * TS + o) * D; const float *v = mid + (b * TS + o) * D;
        for (int ch = c0; ch < c1; ch++) u[ch] = 0;
        for (int c = b + 1; c < NB; c++) {
            float w = Lw[(o * NB + c) * NB + b], g = 0; const float *d = dy + (c * TS + o) * D;
            for (int ch = c0; ch < c1; ch++) { u[ch] += w * d[ch]; g += d[ch] * v[ch]; }
            dL[(o * NB + c) * NB + b] += g;
        }
    }
    for (int b = 0; b < NB; b++) for (int i = 0; i < TS; i++) {
        float *u = dx + (b * TS + i) * D; const float *v = x + (b * TS + i) * D;
        for (int ch = c0; ch < c1; ch++) u[ch] = 0;
        for (int o = 0; o < TS; o++) {
            int k = (b * TS + o) * TS + i; float wr = R[k], wa = A[k], gr = 0, ga = 0;
            const float *dm = dmid + (b * TS + o) * D, *d = dy + (b * TS + o) * D;
            for (int ch = c0; ch < c1; ch++) { u[ch] += wr * dm[ch] + wa * d[ch]; gr += dm[ch] * v[ch]; ga += d[ch] * v[ch]; }
            dR[k] += gr; if (o >= i) dA[k] += ga;
        }
    }
}
static void smon_reduce(SMon *m) {                         // sum slabs; STE: d latent = d effective / sqrt(fan-in)
    #pragma omp parallel for
    for (int k = 0; k < SW; k++) {
        float s = 0; for (int t = 0; t < NT; t++) { s += m->dw[(size_t)t * SW + k]; m->dw[(size_t)t * SW + k] = 0; }
        int f = k < H * SA ? 0 : k < 2 * H * SA ? 1 : 2, j = k - (f == 0 ? 0 : f == 1 ? H * SA : 2 * H * SA), bs = sm_bs(f);
        int fan = sm_fan(f, j % sm_n(f) / bs % bs, j % bs);
        if (fan) m->w[f]->g[j] += s / sqrtf(fan);
    }
}

// ---- model: the M2 layer ---------------------------------------------------------------------------
//   x += M2( K * M1(n) ),       n = rms(x)      sequence mixing: causal T x T Monarchs, K a [T][D] gain
//   x += Mon_d( gelu(Mon_u(n)) ), n = rms(x)      dimension mixing: D x D Monarchs (ternary x int8)
typedef struct {
    SMon m1, m2; P *k; Mon u, d;
    int8_t *q2, *qd;                          // int8 codes: channel Monarch inputs
    float *s2, *sd, *r1, *r2;                 // per-token scales and rms
    float *nx, *mid1, *a, *mid2, *x1, *h;     // float activations kept for backward
} Layer;
static Layer ly[L];
static P *E;                                  // token embedding, tied with the output head (fp32)
static int V;
static float *X[L + 1], *HN, *RF, *LG, *DX, *S1, *BB, *DB, *SCR;   // BB: K * a, DB: its grad, SCR: per-thread [T][D]

static float gelu(float x, float *d) {        // tanh GELU and its derivative; tanh as a clamped [7/6] Pade
    float u = 0.7978846f * (x + 0.044715f * x * x * x), v = fminf(fmaxf(u, -4.97f), 4.97f), v2 = v * v;
    float th = v * (135135 + v2 * (17325 + v2 * (378 + v2))) / (135135 + v2 * (62370 + v2 * (3150 + v2 * 28)));
    *d = 0.5f * (1 + th) + 0.5f * x * (1 - th * th) * 0.7978846f * (1 + 3 * 0.044715f * x * x);
    return 0.5f * x * (1 + th);
}

static void build(void) {
    NT = omp_get_max_threads(); if (NT > 64) NT = 64;
    int lg = 0; while ((1 << lg) < T) lg++;
    TS = 1 << (lg - lg / 2); NB = T / TS; SA = NB * TS * TS; SL = TS * NB * NB;
    E = param(V * D, 0.02f);
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l];
        smon_init(&y->m1); smon_init(&y->m2);
        y->k = param(T * D, 0); for (int i = 0; i < T * D; i++) y->k->w[i] = 1;
        mon_init(&y->u, 1, 10 * l + 1); mon_init(&y->d, 1 / sqrtf(2 * L), 10 * l + 2);
        y->q2 = ia(N * D); y->qd = ia(N * D);
        float **sc[] = {&y->s2, &y->sd, &y->r1, &y->r2};
        for (int i = 0; i < 4; i++) *sc[i] = fa(N);
        float **ac[] = {&y->nx, &y->mid1, &y->a, &y->mid2, &y->x1, &y->h};
        for (int i = 0; i < 6; i++) *ac[i] = fa(N * D);
    }
    for (int l = 0; l <= L; l++) X[l] = fa(N * D);
    HN = fa(N * D); RF = fa(N); DX = fa(N * D); S1 = fa(N * D); LG = fa(N * V); BB = fa(N * D); DB = fa(N * D);
    SCR = fa((size_t)NT * T * D);
}

// sequence mixing forward for one layer: x1 = x + M2(K * M1(rms x))
static void seqmix_fwd(Layer *y, const float *x, int nb) {
    int n = nb * T;
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {           // n = rms(x)
        const float *xt = x + t * D; float r = y->r1[t] = rinv(xt);
        for (int i = 0; i < D; i++) y->nx[t * D + i] = xt[i] * r;
    }
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < nb; b++) for (int c0 = 0; c0 < D; c0 += CW) {   // sequence mixing, per sequence and channel group
        int o = b * T * D, h = c0 / HD; float *s = SCR + (size_t)TID * T * D;
        smon_fwd(&y->m1, y->nx + o, y->mid1 + o, y->a + o, h, c0, c0 + CW);
        for (int t = 0; t < T; t++) for (int ch = c0; ch < c0 + CW; ch++) BB[o + t * D + ch] = y->k->w[t * D + ch] * y->a[o + t * D + ch];
        smon_fwd(&y->m2, BB + o, y->mid2 + o, s, h, c0, c0 + CW);
        for (int t = 0; t < T; t++) for (int ch = c0; ch < c0 + CW; ch++) y->x1[o + t * D + ch] = x[o + t * D + ch] + s[t * D + ch];
    }
}

static void forward(const int *tok, int nb) {
    int n = nb * T;
    for (int t = 0; t < n; t++) memcpy(X[0] + t * D, E->w + tok[t] * D, D * sizeof(float));
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l];
        smon_prep(&y->m1); smon_prep(&y->m2); mon_prep(&y->u); mon_prep(&y->d);
        seqmix_fwd(y, X[l], nb);
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {           // dimension mixing: rms, Mon_u, gelu, Mon_d, +res
            float g[D], o[D], dd, *x1 = y->x1 + t * D, *h = y->h + t * D;
            y->r2[t] = rinv(x1); y->s2[t] = q8t(x1, y->q2 + t * D, y->r2[t]);
            mon_fwd(&y->u, y->q2 + t * D, y->s2[t], h, t);
            #pragma omp simd private(dd)
            for (int i = 0; i < D; i++) g[i] = gelu(h[i], &dd);
            y->sd[t] = q8t(g, y->qd + t * D, 1);
            mon_fwd(&y->d, y->qd + t * D, y->sd[t], o, t);
            for (int i = 0; i < D; i++) X[l + 1][t * D + i] = x1[i] + o[i];
        }
    }
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {
        float r = RF[t] = rinv(X[L] + t * D);
        for (int i = 0; i < D; i++) HN[t * D + i] = X[L][t * D + i] * r;
        for (int c = 0; c < V; c++) {
            float s = 0; for (int i = 0; i < D; i++) s += HN[t * D + i] * E->w[c * D + i];
            LG[t * V + c] = s;
        }
    }
}

static float xent(const int *tgt, int n, int grad) {   // mean CE; if grad, LG <- dLoss/dlogits
    double loss = 0;
    for (int t = 0; t < n; t++) {
        float *z = LG + t * V, mx = z[0], s = 0;
        for (int c = 1; c < V; c++) mx = fmaxf(mx, z[c]);
        for (int c = 0; c < V; c++) s += expf(z[c] - mx);
        loss += logf(s) + mx - z[tgt[t]];
        if (grad) { for (int c = 0; c < V; c++) z[c] = expf(z[c] - mx) / s / n; z[tgt[t]] -= 1.f / n; }
    }
    return loss / n;
}

// sequence mixing backward for one layer: DX (grad wrt x1) -> DX += grad through M2(K * M1(rms x))
static void seqmix_bwd(Layer *y, const float *x, int nb) {
    int n = nb * T;
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < nb; b++) for (int c0 = 0; c0 < D; c0 += CW) {       // M2^T: dS -> dB
        int o = b * T * D, h = c0 / HD; float *s = SCR + (size_t)TID * T * D, *dw = y->m2.dw + (size_t)TID * SW;
        smon_bwd(&y->m2, BB + o, y->mid2 + o, DX + o, s, DB + o, h, c0, c0 + CW, dw);
    }
    #pragma omp parallel for
    for (int t = 0; t < T; t++) for (int ch = 0; ch < D; ch++) {            // dK = sum_b dB a
        float g = 0; for (int b = 0; b < nb; b++) g += DB[(b * T + t) * D + ch] * y->a[(b * T + t) * D + ch];
        y->k->g[t * D + ch] += g;
    }
    #pragma omp parallel for collapse(2)
    for (int b = 0; b < nb; b++) for (int c0 = 0; c0 < D; c0 += CW) {       // dA = dB K; M1^T: dA -> dn (into BB)
        int o = b * T * D, h = c0 / HD; float *s = SCR + (size_t)TID * T * D, *dw = y->m1.dw + (size_t)TID * SW;
        for (int t = 0; t < T; t++) for (int ch = c0; ch < c0 + CW; ch++) DB[o + t * D + ch] *= y->k->w[t * D + ch];
        smon_bwd(&y->m1, y->nx + o, y->mid1 + o, DB + o, s, BB + o, h, c0, c0 + CW, dw);
    }
    #pragma omp parallel for
    for (int t = 0; t < n; t++) rms_bwd(x + t * D, y->r1[t], BB + t * D, DX + t * D);
    smon_reduce(&y->m1); smon_reduce(&y->m2);
}

static void backward(const int *tok, int nb) {
    int n = nb * T;
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < D; i++) {
            float s = 0; for (int c = 0; c < V; c++) s += LG[t * V + c] * E->w[c * D + i];
            S1[t * D + i] = s; DX[t * D + i] = 0;
        }
        rms_bwd(X[L] + t * D, RF[t], S1 + t * D, DX + t * D);
    }
    #pragma omp parallel for
    for (int c = 0; c < V; c++) for (int t = 0; t < n; t++) for (int i = 0; i < D; i++)
        E->g[c * D + i] += LG[t * V + c] * HN[t * D + i];
    for (int l = L - 1; l >= 0; l--) {                  // DX: grad wrt X[l+1] -> grad wrt X[l]
        Layer *y = &ly[l];
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {                   // dimension mixing
            int tid = TID; float a[D], b[D], dd, *dx = DX + t * D, *h = y->h + t * D;
            mon_bwd(&y->d, y->qd + t * D, y->sd[t], dx, a, t, tid);
            #pragma omp simd private(dd)
            for (int i = 0; i < D; i++) { gelu(h[i], &dd); a[i] *= dd; }
            mon_bwd(&y->u, y->q2 + t * D, y->s2[t], a, b, t, tid);
            rms_bwd(y->x1 + t * D, y->r2[t], b, dx);
        }
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < nb; b++) for (int c0 = 0; c0 < D; c0 += CW)    // B = K * a, for M2's weight grads
            for (int t = 0; t < T; t++) for (int ch = c0; ch < c0 + CW; ch++)
                BB[(b * T + t) * D + ch] = y->k->w[t * D + ch] * y->a[(b * T + t) * D + ch];
        seqmix_bwd(y, X[l], nb);
        mon_reduce(&y->u); mon_reduce(&y->d);
    }
    for (int t = 0; t < n; t++) for (int i = 0; i < D; i++) E->g[tok[t] * D + i] += DX[t * D + i];
}

static void adam(int step) {
    float lr = step < 100 ? LR * step / 100 : LR * (0.1f + 0.45f * (1 + cosf(3.14159265f * step / STEPS)));
    float c1 = 1 - powf(0.9f, step), c2 = 1 - powf(0.95f, step);
    for (int k = 0; k < np; k++) {
        P *p = &ps[k];
        for (int i = 0; i < p->n; i++) {
            float g = p->g[i]; p->g[i] = 0;
            if (step <= 0) continue;
            p->m[i] = 0.9f * p->m[i] + 0.1f * g; p->v[i] = 0.95f * p->v[i] + 0.05f * g * g;
            p->w[i] -= lr * (p->m[i] / c1) / (sqrtf(p->v[i] / c2) + 1e-8f);
        }
    }
}

static int *data, ndata, ntrain;
static void batch(int *tok, int *tgt, int val) {
    for (int b = 0; b < B; b++) {
        int lo = val ? ntrain : 0, hi = val ? ndata : ntrain;
        int off = lo + (int)(urand() * (hi - lo - T - 1));
        for (int t = 0; t < T; t++) { tok[b * T + t] = data[off + t]; tgt[b * T + t] = data[off + t + 1]; }
    }
}
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
    FILE *f = fopen(argc > 1 ? argv[1] : "input.txt", "rb");
    if (!f) f = fopen(__FILE__, "rb");                  // no data? learn to write moth.c
    if (!f) { fprintf(stderr, "no input\n"); return 1; }
    fseek(f, 0, SEEK_END); ndata = ftell(f); rewind(f);
    unsigned char *raw = malloc(ndata); if (fread(raw, 1, ndata, f) != (size_t)ndata) return 1; fclose(f);
    int stoi[256], itos[256]; memset(stoi, -1, sizeof stoi);
    for (int i = 0; i < ndata; i++) if (stoi[raw[i]] < 0) stoi[raw[i]] = 0;
    for (int c = 0; c < 256; c++) if (!stoi[c]) { itos[V] = c; stoi[c] = V++; }
    data = malloc(ndata * sizeof(int)); for (int i = 0; i < ndata; i++) data[i] = stoi[raw[i]];
    ntrain = ndata * 9 / 10;

    build();
    long nseq = NB * TS * (TS + 1) / 2 + SA + TS * NB * (NB - 1) / 2, ntern = L * (2 * 2 * W3 + 2 * H * nseq), nparam = 0;
    for (int k = 0; k < np; k++) nparam += ps[k].n;
    nparam -= L * 2 * H * (2 * SA + SL - nseq);             // the causal masks' structural zeros
    printf("moth: vocab %d, d %d, layers %d, T %d, %.2fM params (%.2fM ternary), %d threads\n",
           V, D, L, T, nparam / 1e6, ntern / 1e6, NT);

    int *tok = malloc(N * sizeof(int)), *tgt = malloc(N * sizeof(int));
    double t0 = now();
    for (int step = -1; step <= STEPS; step++) {        // steps -1, 0 calibrate the delayed scales
        seed = hash(step + 2);
        batch(tok, tgt, 0);
        forward(tok, B);
        float loss = xent(tgt, N, 1);
        backward(tok, B);
        adam(step);
        if (step % 50 == 0 && step > 0) { printf("step %4d | loss %.4f | %.0f ms/step\n", step, loss, (now() - t0) * 1e3 / 50); fflush(stdout); t0 = now(); }
        if (step % 500 == 0 && step > 0) {
            float vl = 0; for (int k = 0; k < 8; k++) { batch(tok, tgt, 1); forward(tok, B); vl += xent(tgt, N, 0) / 8; }
            printf("step %4d | val loss %.4f\n", step, vl); t0 = now();
        }
    }

    int ctx[T] = {0}, len = 1; ctx[0] = data[0];           // sample (recomputes the window per token)
    putchar(itos[ctx[0]]);
    for (int k = 0; k < 600; k++) {
        forward(ctx, 1);
        float *z = LG + (len - 1) * V, mx = -1e30f, s = 0, r;
        for (int c = 0; c < V; c++) mx = fmaxf(mx, z[c]);
        for (int c = 0; c < V; c++) s += (z[c] = expf((z[c] - mx) / 0.8f));
        int c = 0; r = urand() * s; while (c < V - 1 && (r -= z[c]) > 0) c++;
        putchar(itos[c]); fflush(stdout);
        if (len == T) { memmove(ctx, ctx + 1, (T - 1) * sizeof(int)); len--; }
        ctx[len++] = c;
    }
    putchar('\n');
    return 0;
}
