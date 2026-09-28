// moth.c: Moth, a tiny ternary Monarch Mixer char-level LM in one file of pure C (nanoGPT style).
//
//   block(x) (M2: Monarch Mixer, Fu et al. 2023, in its causal / GPT form):
//     n = rms(x);  q, k, v = sconv3(Mon_q n), sconv3(Mon_k n), sconv3(Mon_v n)   sconv3: depthwise causal, width 3
//     x += Mon_o( q * (h conv u + bias * u) ),  u = k * v                        sequence mixer (order-2 gated conv)
//     n = rms(x);  x += Mon_d( gelu(Mon_g n) * Mon_u n )                         channel mixer (Monarch GLU)
//   Mon: view a width-d vector as an MxM tile (d = M^2). R mixes along each row, L along each column.
//        That is P L P^T R. The permutations are never materialised: a column op on a row-major tile is
//        lane-wise vector math with weights stored [o][i][lane], so every loop streams contiguous int8.
//   conv: the length-T causal long conv is an FFT conv of length 2T, and the DFT is itself a Monarch of dense
//        blocks (block DFT matmuls, twiddle, block DFT matmuls), O(T^1.5 D) per sequence instead of O(T^2 D).
//        The filter h is implicit: a small sine FFN of positional features, times a per-channel exp decay.
//        It only exists in training; what the model runs on is its ternarisation.
//
// forward : every weight inference touches is a trit {-1,0,1} with an absmean scale, trained with STE:
//           the Monarchs, the long-conv kernel (per-channel scale), the short convs and both biases. Monarch
//           inputs are int8 per-token absmax codes; a ternary dot over 16 of them fits in int16, so those
//           accumulators use 32 lanes, not 16. Absmax codes are scale-invariant, so RMSNorm, R->L requantisation
//           and the gates fold into per-token scales. Signals mixed across tokens (q, k, v into the short convs,
//           u into the long conv) are int8 on a static EMA scale instead, so a sum over tokens stays integer.
//           Training runs the long conv as an FFT over these fake-quantised values; inference (infer_step)
//           runs it as a direct trit x int8 sum, one token at a time, with no FFT and no window recompute.
// backward: gradients are int8 with a delayed per-tensor scale (last step's amax, as in FP8 training) and
//           stochastic rounding. There is no amax pre-pass and no barrier inside a Monarch. The input's per-token
//           scale is folded into the gradient, so dW = g8^T x8 reuses the forward's int8 codes.
//           dx = W^T g8 (ternary x int8, int16) and dW (int8 x int8) accumulate in per-thread int32 slabs.
//           The long conv's backward is two more FFT correlations: du = g corr h, dh = sum_b g corr u.
//
// cc -O3 -march=native -fopenmp moth.c -o moth -lm && ./moth input.txt
// (on AVX-512 add -mprefer-vector-width=512: inference runs ~25% faster, training is unchanged)
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
static P ps[1 + 22 * L]; static int np;   // embedding + 22 tensors per layer
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
    float m = 1e-8f;
    #pragma omp simd reduction(max:m)
    for (int i = 0; i < D; i++) { float a = fabsf(x[i]); m = a > m ? a : m; }
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
static float rinv(const float *x) {
    float s = 0;
    #pragma omp simd reduction(+:s)
    for (int i = 0; i < D; i++) s += x[i] * x[i];
    return 1 / sqrtf(s / D + 1e-5f);
}
static void rms_bwd(const float *x, float r, const float *dy, float *dx) {   // dx += d rms(x)
    float dot = 0; for (int i = 0; i < D; i++) dot += dy[i] * x[i]; dot *= r * r / D;
    for (int i = 0; i < D; i++) dx[i] += (dy[i] - x[i] * dot) * r;
}

// ---- Monarch FFT: the long conv as F^-1 (F k . F v), zero-padded to NF = 2T so it is causal ---------
// As in M2 / FlashFFTConv, the DFT is a Monarch of dense blocks, F_NF = P (I (x) F_P1) P^T Tw (I (x) F_P2) P
// with NF = P1 P2 (16 x 16 at T = 128): view the first T steps as a P1/2 x P2 tile (the other half is the
// causal zero padding), multiply by the dense P1 x P1 DFT down its columns, twiddle, then by the dense
// P2 x P2 DFT along its rows. Both are block matmuls, lane-wise over CH channels, like Mon's R and L. The
// permutations are never materialised: bin k1 + P1 k2 sits at [k1][k2], and everything done to a spectrum
// is pointwise. Two real sequences ride in one complex transform (re, im); a real kernel keeps them apart.
#define NF (2 * T)
#define CH 16                                 // channels per vector lane group
#define NCH (D / CH)
static int P1, P2;
static float *F1r, *F1i, *F2r, *F2i, *TWr, *TWi, *SCR;   // dense block DFTs, twiddle [k1][n2], scratch
static void fft_init(void) {
    int lg = 0; while ((1 << lg) < NF) lg++;
    P1 = 1 << lg / 2; P2 = NF / P1;
    F1r = fa(P1 * P1); F1i = fa(P1 * P1); F2r = fa(P2 * P2); F2i = fa(P2 * P2); TWr = fa(NF); TWi = fa(NF);
    for (int k = 0; k < P1 * P1; k++) { F1r[k] = cos(2 * M_PI * (k / P1) * (k % P1) / P1); F1i[k] = -sin(2 * M_PI * (k / P1) * (k % P1) / P1); }
    for (int k = 0; k < P2 * P2; k++) { F2r[k] = cos(2 * M_PI * (k / P2) * (k % P2) / P2); F2i[k] = -sin(2 * M_PI * (k / P2) * (k % P2) / P2); }
    for (int k = 0; k < NF; k++) { TWr[k] = cos(2 * M_PI * (k / P2) * (k % P2) / NF); TWi[k] = -sin(2 * M_PI * (k / P2) * (k % P2) / NF); }
    SCR = fa((size_t)NT * 6 * NF * CH);
}
// dense block DFT: out[k] = sum_n (c + i s d) F[k][n] in[n], each entry a vector of w floats (s = -1: inverse).
// Accumulates CH lanes at a time in registers; in and out must not overlap.
static void bdft(const float *restrict ir, const float *restrict ii, float *restrict or, float *restrict oi,
                 int nk, int nn, int ld, const float *Fr, const float *Fi, int w, float s) {
    for (int k = 0; k < nk; k++) for (int j0 = 0; j0 < w; j0 += CH) {
        float ur[CH] = {0}, ui[CH] = {0};
        for (int n = 0; n < nn; n++) {
            float c = Fr[k * ld + n], d = s * Fi[k * ld + n]; const float *xr = ir + n * w + j0, *xi = ii + n * w + j0;
            #pragma omp simd simdlen(CH)
            for (int l = 0; l < CH; l++) { ur[l] += c * xr[l] - d * xi[l]; ui[l] += c * xi[l] + d * xr[l]; }
        }
        memcpy(or + k * w + j0, ur, CH * 4); memcpy(oi + k * w + j0, ui, CH * 4);
    }
}
static void twid(float *re, float *im, float s, float sg) {       // [k1][n2][CH] *= s TW^sg, sg = +-1
    for (int k = 0; k < NF; k++) {
        float c = TWr[k] * s, d = sg * TWi[k] * s, *a = re + k * CH, *b = im + k * CH;
        for (int l = 0; l < CH; l++) { float r = a[l] * c - b[l] * d; b[l] = a[l] * d + b[l] * c; a[l] = r; }
    }
}
// x (+ i y), T steps of CH lanes at stride xs (y may be NULL) -> spectrum (or, oi) [P1][P2][CH]
static void mfft(const float *x, const float *y, int xs, float *or, float *oi, float *tr, float *ti) {
    int R = P2 * CH; float *rr = tr + T * CH, *ri = ti + T * CH;
    for (int t = 0; t < T; t++) for (int l = 0; l < CH; l++) { tr[t * CH + l] = x[t * xs + l]; ti[t * CH + l] = y ? y[t * xs + l] : 0; }
    bdft(tr, ti, or, oi, P1, P1 / 2, P1, F1r, F1i, R, 1);        // columns: P1 x P1/2 (zero half skipped)
    twid(or, oi, 1, 1);
    for (int k1 = 0; k1 < P1; k1++) {                              // rows: P2 x P2
        memcpy(rr, or + k1 * R, R * 4); memcpy(ri, oi + k1 * R, R * 4);
        bdft(rr, ri, or + k1 * R, oi + k1 * R, P2, P2, P2, F2r, F2i, CH, 1);
    }
}
// spectrum (zr, zi) (destroyed) -> first T steps: real part to x, imaginary part to y (if not NULL), times s
static void mifft(float *zr, float *zi, float *x, float *y, int xs, float s, float *tr, float *ti) {
    int R = P2 * CH;
    for (int k1 = 0; k1 < P1; k1++) {                              // rows: F_P2^-1
        memcpy(tr, zr + k1 * R, R * 4); memcpy(ti, zi + k1 * R, R * 4);
        bdft(tr, ti, zr + k1 * R, zi + k1 * R, P2, P2, P2, F2r, F2i, CH, -1);
    }
    twid(zr, zi, s, -1);
    bdft(zr, zi, tr, ti, P1 / 2, P1, P1, F1r, F1i, R, -1);        // columns: only the P1/2 causal outputs
    for (int t = 0; t < T; t++) { memcpy(x + t * xs, tr + t * CH, CH * 4); if (y) memcpy(y + t * xs, ti + t * CH, CH * 4); }
}

// ---- model -------------------------------------------------------------------------------------
// Sequence mixer (M2's causal mixer, the Hyena order-2 operator on the Monarch FFT):
//   q, k, v = shortconv3(Mon_q n), shortconv3(Mon_k n), shortconv3(Mon_v n)     depthwise causal, width 3
//   y = q * (h conv u + bias * u),  u = k * v
//   h[t] = FFN(pos(t)) * exp(-|a_c| t / T)     implicit filter: a sine FFN on positional features, decayed
// Channel mixer (M2's Monarch GLU): Mon_d( gelu(Mon_g n) * Mon_u n )
#define FE 17                                 // filter positional features: t, 8 cos/sin bands
#define FO 64                                 // filter FFN width
typedef struct {
    P *w1, *b1, *w2, *b2, *w3, *bias, *sw, *sb;           // filter FFN, skip bias [D], short conv [3][3D] + [3D]
    float *pz, *mod, *p1, *a1, *p2, *a2, *h, *dh, *Kr, *Ki, *Zr, *Zi;
    int8_t hq[T * D], swq[9 * D], sbq[3 * D], bq[D];      // trits: long kernel [T][D], short conv, its bias, skip bias
    float hs[D], sws[3], sbs[3], bs;                      // their absmean scales (kernel: per channel; rest: per stream)
    float he[T * D], swe[9 * D], sbe[3 * D], be[D];       // trit * scale, what training runs on
    float amax[4], as[4];                                 // static int8 scales for pre q, k, v and u: EMA amax, amax / 127
} Conv;
typedef struct {
    Mon q, k, v, o, g, u, d; Conv cv;
    int8_t *q1, *qo, *q2, *qd;                // int8 codes: Monarch inputs
    float *s1, *so, *s2, *sd, *r1, *r2;       // per-token scales and rms
    float *pre, *post, *cc, *x1, *ga, *ub;    // float activations kept for backward (pre/post: [3][N][D])
} Layer;
static Layer ly[L];
static P *E;                                  // token embedding, tied with the output head (fp32)
static int V;
static float *X[L + 1], *HN, *RF, *LG, *DX, *S1, *VB, *DP;   // VB: conv input (grad); DP: short conv out grads

static inline float tanh_(float u) {           // tanh as a clamped [7/6] Pade
    float v = fminf(fmaxf(u, -4.97f), 4.97f), v2 = v * v;
    return v * (135135 + v2 * (17325 + v2 * (378 + v2))) / (135135 + v2 * (62370 + v2 * (3150 + v2 * 28)));
}
static inline float gelu0(float x) { return 0.5f * x * (1 + tanh_(0.7978846f * (x + 0.044715f * x * x * x))); }
static float gelu(float x, float *d) {        // tanh GELU and its derivative
    float u = 0.7978846f * (x + 0.044715f * x * x * x), th = tanh_(u);
    *d = 0.5f * (1 + th) + 0.5f * x * (1 - th * th) * 0.7978846f * (1 + 3 * 0.044715f * x * x);
    return 0.5f * x * (1 + th);
}

static void build(void) {
    NT = omp_get_max_threads(); if (NT > 64) NT = 64;
    fft_init();
    E = param(V * D, 0.02f);
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; Conv *c = &y->cv; float ro = 1 / sqrtf(2 * L);
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_init(ms[i], ms[i] == &y->o || ms[i] == &y->d ? ro : 1, 10 * l + i + 1);
        c->w1 = param(FO * FE, 1 / sqrtf(FE)); c->b1 = param(FO, 1 / sqrtf(FE));
        c->w2 = param(FO * FO, 1 / sqrtf(FO)); c->b2 = param(FO, 1 / sqrtf(FO)); c->w3 = param(D * FO, 1 / sqrtf(FO));
        c->bias = param(D, 1); c->sw = param(3 * 3 * D, 1 / 3.f); c->sb = param(3 * D, 1 / 3.f);
        c->pz = fa(T * FE); c->mod = fa(T * D);
        for (int t = 0; t < T; t++) {
            float tl = (float)t / (T - 1); c->pz[t * FE] = tl;
            for (int f = 0; f < 8; f++) {
                float fr = 1e-4f + f * (7 - 1e-4f) / 7;
                c->pz[t * FE + 1 + f] = cosf(2 * M_PI * fr * t / T); c->pz[t * FE + 9 + f] = -sinf(2 * M_PI * fr * t / T);
            }
            for (int ch = 0; ch < D; ch++) {      // decay rates spread from slow (1% at t = 1.5T) to fast (0.3T)
                float lo = logf(1e-2f) / 1.5f, hi = logf(1e-2f) / 0.3f;
                c->mod[t * D + ch] = expf(-tl * fabsf(lo + (hi - lo) * ch / (D - 1)));
            }
        }
        c->p1 = fa(T * FO); c->a1 = fa(T * FO); c->p2 = fa(T * FO); c->a2 = fa(T * FO); c->h = fa(T * D); c->dh = fa(T * D);
        c->Kr = fa(NF * D); c->Ki = fa(NF * D); c->Zr = fa((size_t)(B + 1) / 2 * NF * D); c->Zi = fa((size_t)(B + 1) / 2 * NF * D);
        int8_t **q[] = {&y->q1, &y->qo, &y->q2, &y->qd};
        for (int i = 0; i < 4; i++) *q[i] = ia(N * D);
        float **sc[] = {&y->s1, &y->so, &y->s2, &y->sd, &y->r1, &y->r2};
        for (int i = 0; i < 6; i++) *sc[i] = fa(N);
        y->pre = fa(3 * N * D); y->post = fa(3 * N * D); y->cc = fa(N * D); y->x1 = fa(N * D); y->ga = fa(N * D); y->ub = fa(N * D);
    }
    for (int l = 0; l <= L; l++) X[l] = fa(N * D);
    HN = fa(N * D); RF = fa(N); VB = fa(N * D); DP = fa(3 * N * D); DX = fa(N * D); S1 = fa(N * D); LG = fa(N * V);
}

static void conv_prep(Conv *c) {                      // ternarise short conv (per stream) and both biases
    for (int w = 0; w < 3; w++) {
        float s = 1e-8f; for (int j = 0; j < 3; j++) for (int ch = 0; ch < D; ch++) s += fabsf(c->sw->w[j * 3 * D + w * D + ch]);
        s /= 3 * D; c->sws[w] = s;
        for (int j = 0; j < 3; j++) for (int ch = 0; ch < D; ch++) {
            int k = j * 3 * D + w * D + ch; float r = roundf(c->sw->w[k] / s);
            c->swq[k] = r > 1 ? 1 : r < -1 ? -1 : r; c->swe[k] = c->swq[k] * s;
        }
        c->sbs[w] = tern(c->sb->w + w * D, c->sbq + w * D, D, 1);
        for (int ch = 0; ch < D; ch++) c->sbe[w * D + ch] = c->sbq[w * D + ch] * c->sbs[w];
    }
    c->bs = tern(c->bias->w, c->bq, D, 1);
    for (int ch = 0; ch < D; ch++) c->be[ch] = c->bq[ch] * c->bs;
    for (int i = 0; i < 4; i++) c->as[i] = (c->amax[i] > 0 ? c->amax[i] : 1) / 127;
}
static int8_t sc8(float x, float is) { float r = rintf(x * is); return r > 127 ? 127 : r < -127 ? -127 : (int8_t)r; }   // is: 1 / scale
// static-scale int8 fake quant (STE): x <- s clip(round(x / s)); returns max |x| for the EMA
static float sq8(float *x, int n, float s) {
    float m = 0, is = 1 / s; for (int i = 0; i < n; i++) { m = fmaxf(m, fabsf(x[i])); x[i] = sc8(x[i], is) * s; }
    return m;
}
static void filter_fwd(Conv *c) {                     // h = FFN(pos) * mod, ternarised per channel, and its spectrum
    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        float *p1 = c->p1 + t * FO, *a1 = c->a1 + t * FO, *p2 = c->p2 + t * FO, *a2 = c->a2 + t * FO;
        for (int o = 0; o < FO; o++) {
            float s = c->b1->w[o]; for (int e = 0; e < FE; e++) s += c->w1->w[o * FE + e] * c->pz[t * FE + e];
            p1[o] = s; a1[o] = sinf(s);
        }
        for (int o = 0; o < FO; o++) {
            float s = c->b2->w[o]; for (int i = 0; i < FO; i++) s += c->w2->w[o * FO + i] * a1[i];
            p2[o] = s; a2[o] = sinf(s);
        }
        for (int ch = 0; ch < D; ch++) {
            float s = 0; for (int o = 0; o < FO; o++) s += c->w3->w[ch * FO + o] * a2[o];
            c->h[t * D + ch] = s * c->mod[t * D + ch];
        }
    }
    for (int ch = 0; ch < D; ch++) {                  // the kernel inference sees: trits, one scale per channel
        c->hs[ch] = tern(c->h + ch, c->hq + ch, T, D);
        for (int t = 0; t < T; t++) c->he[t * D + ch] = c->hq[t * D + ch] * c->hs[ch];
    }
    #pragma omp parallel for
    for (int ch = 0; ch < NCH; ch++) {
        float *s = SCR + (size_t)TID * 6 * NF * CH;
        mfft(c->he + ch * CH, NULL, D, c->Kr + ch * NF * CH, c->Ki + ch * NF * CH, s, s + NF * CH);
    }
}
static void filter_bwd(Conv *c) {                     // dh -> filter FFN grads
    float *g2 = fa(T * FO), *g1 = fa(T * FO);
    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        float s[FO] = {0};
        for (int ch = 0; ch < D; ch++) {
            float g = c->dh[t * D + ch] *= c->mod[t * D + ch]; const float *w = c->w3->w + ch * FO;
            for (int o = 0; o < FO; o++) s[o] += g * w[o];
        }
        for (int o = 0; o < FO; o++) g2[t * FO + o] = s[o] * cosf(c->p2[t * FO + o]);
        for (int i = 0; i < FO; i++) {
            float s = 0; for (int o = 0; o < FO; o++) s += g2[t * FO + o] * c->w2->w[o * FO + i];
            g1[t * FO + i] = s * cosf(c->p1[t * FO + i]);
        }
    }
    #pragma omp parallel for
    for (int ch = 0; ch < D; ch++) for (int t = 0; t < T; t++) {
        float g = c->dh[t * D + ch], *w = c->w3->g + ch * FO; const float *a = c->a2 + t * FO;
        for (int o = 0; o < FO; o++) w[o] += g * a[o];
    }
    #pragma omp parallel for
    for (int o = 0; o < FO; o++) {
        for (int t = 0; t < T; t++) {
            c->b2->g[o] += g2[t * FO + o]; c->b1->g[o] += g1[t * FO + o];
            for (int i = 0; i < FO; i++) c->w2->g[o * FO + i] += g2[t * FO + o] * c->a1[t * FO + i];
            for (int e = 0; e < FE; e++) c->w1->g[o * FE + e] += g1[t * FO + o] * c->pz[t * FE + e];
        }
    }
    free(g2); free(g1);
}

// pre (Mon_q/k/v outputs) -> short convs -> post, u = k * v -> VB, long conv h * u -> cc
static void mixer_fwd(Layer *y, int nb, int train) {
    Conv *c = &y->cv; int n = nb * T, np2 = (nb + 1) / 2; float m0 = 0, m1 = 0, m2 = 0, m3 = 0;
    filter_fwd(c); conv_prep(c);
    #pragma omp parallel for reduction(max:m0, m1, m2)
    for (int t = 0; t < n; t++) {           // q, k, v to int8 codes on a static scale: they're mixed across tokens
        m0 = fmaxf(m0, sq8(y->pre + (size_t)t * D, D, c->as[0]));
        m1 = fmaxf(m1, sq8(y->pre + ((size_t)N + t) * D, D, c->as[1]));
        m2 = fmaxf(m2, sq8(y->pre + ((size_t)2 * N + t) * D, D, c->as[2]));
    }
    #pragma omp parallel for reduction(max:m3)
    for (int t = 0; t < n; t++) {           // ternary short convs; u = k * v, int8 on a static scale, feeds the long conv
        int tt = t % T;
        for (int w = 0; w < 3; w++) {
            float *o = y->post + ((size_t)w * N + t) * D; const float *sb = c->sbe + w * D;
            for (int ch = 0; ch < D; ch++) o[ch] = sb[ch];
            for (int j = 0; j < 3 && j <= tt; j++) {
                const float *sw = c->swe + j * 3 * D + w * D, *p = y->pre + ((size_t)w * N + t - j) * D;
                for (int ch = 0; ch < D; ch++) o[ch] += sw[ch] * p[ch];
            }
        }
        const float *k = y->post + ((size_t)N + t) * D, *v = y->post + ((size_t)2 * N + t) * D;
        for (int ch = 0; ch < D; ch++) VB[t * D + ch] = k[ch] * v[ch];
        m3 = fmaxf(m3, sq8(VB + t * D, D, c->as[3]));
    }
    if (train) {                            // EMA of amax, updated after use so the scale stays causal
        float m[4] = {m0, m1, m2, m3};
        for (int i = 0; i < 4; i++) c->amax[i] = c->amax[i] > 0 ? 0.99f * c->amax[i] + 0.01f * m[i] : m[i];
    }
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < np2; p++) for (int ch = 0; ch < NCH; ch++) {   // long conv, two sequences at a time
        float *s = SCR + (size_t)TID * 6 * NF * CH, *ur = s + 2 * NF * CH, *ui = ur + NF * CH;
        float *zr = c->Zr + (size_t)(p * NCH + ch) * NF * CH, *zi = c->Zi + (size_t)(p * NCH + ch) * NF * CH;
        const float *kr = c->Kr + ch * NF * CH, *ki = c->Ki + ch * NF * CH;
        int o0 = 2 * p * T * D + ch * CH, two = 2 * p + 1 < nb;
        mfft(VB + o0, two ? VB + o0 + T * D : NULL, D, zr, zi, s, s + NF * CH);
        for (int k = 0; k < NF * CH; k++) { ur[k] = zr[k] * kr[k] - zi[k] * ki[k]; ui[k] = zr[k] * ki[k] + zi[k] * kr[k]; }
        mifft(ur, ui, y->cc + o0, two ? y->cc + o0 + T * D : NULL, D, 1.f / NF, s, s + NF * CH);
    }
}

// DP[0] = d q (post), cc = d(conv out) -> pre <- d pre, and the mixer's parameter grads
static void mixer_bwd(Layer *y, int nb) {
    Conv *c = &y->cv; int n = nb * T, np2 = (nb + 1) / 2;
    #pragma omp parallel for
    for (int ch = 0; ch < NCH; ch++) {             // conv^T: du = g corr h, dh = sum_b g corr u (Re of the
        float *s = SCR + (size_t)TID * 6 * NF * CH, *gr = s + 2 * NF * CH, *gi = gr + NF * CH, *ar = gi + NF * CH, *ai = ar + NF * CH;
        const float *kr = c->Kr + ch * NF * CH, *ki = c->Ki + ch * NF * CH;      // packed product: the cross
        memset(ar, 0, 2 * NF * CH * 4);                                          // terms are imaginary)
        for (int p = 0; p < np2; p++) {
            const float *zr = c->Zr + (size_t)(p * NCH + ch) * NF * CH, *zi = c->Zi + (size_t)(p * NCH + ch) * NF * CH;
            int o0 = 2 * p * T * D + ch * CH, two = 2 * p + 1 < nb;
            mfft(y->cc + o0, two ? y->cc + o0 + T * D : NULL, D, gr, gi, s, s + NF * CH);
            for (int k = 0; k < NF * CH; k++) {
                ar[k] += gr[k] * zr[k] + gi[k] * zi[k]; ai[k] += gi[k] * zr[k] - gr[k] * zi[k];
                float r = gr[k] * kr[k] + gi[k] * ki[k]; gi[k] = gi[k] * kr[k] - gr[k] * ki[k]; gr[k] = r;
            }
            mifft(gr, gi, VB + o0, two ? VB + o0 + T * D : NULL, D, 1.f / NF, s, s + NF * CH);
        }
        mifft(ar, ai, c->dh + ch * CH, NULL, D, 1.f / NF, s, s + NF * CH);
    }
    filter_bwd(c);
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {                   // du (+ skip) -> dk, dv
        float *cc = y->cc + t * D, *k = y->post + ((size_t)N + t) * D, *v = k + (size_t)N * D;
        for (int ch = 0; ch < D; ch++) {
            float du = VB[t * D + ch] + cc[ch] * c->be[ch];
            DP[((size_t)N + t) * D + ch] = du * v[ch]; DP[((size_t)2 * N + t) * D + ch] = du * k[ch];
        }
    }
    #pragma omp parallel for
    for (int c0 = 0; c0 < D; c0 += CH) {            // skip bias and short conv weight grads, CH lanes at a time
        float gb[CH] = {0}, gw[9][CH] = {{0}}, gs[3][CH] = {{0}};
        for (int t = 0; t < n; t++) {
            int tt = t % T; const float *k = y->post + ((size_t)N + t) * D + c0, *v = k + (size_t)N * D, *cc = y->cc + t * D + c0;
            for (int l = 0; l < CH; l++) gb[l] += cc[l] * sc8(k[l] * v[l], 1 / c->as[3]) * c->as[3];
            for (int w = 0; w < 3; w++) {
                const float *g = DP + ((size_t)w * N + t) * D + c0;
                for (int l = 0; l < CH; l++) gs[w][l] += g[l];
                for (int j = 0; j < 3 && j <= tt; j++) {
                    const float *p = y->pre + ((size_t)w * N + t - j) * D + c0;
                    for (int l = 0; l < CH; l++) gw[j * 3 + w][l] += g[l] * p[l];
                }
            }
        }
        for (int l = 0; l < CH; l++) {
            c->bias->g[c0 + l] += gb[l];
            for (int w = 0; w < 3; w++) { c->sb->g[w * D + c0 + l] += gs[w][l]; for (int j = 0; j < 3; j++) c->sw->g[j * 3 * D + w * D + c0 + l] += gw[j * 3 + w][l]; }
        }
    }
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {                       // short conv^T: d pre[t] = sum_j w_j d post[t + j]
        int tt = t % T;
        for (int w = 0; w < 3; w++) {
            float *a = y->pre + ((size_t)w * N + t) * D;
            for (int ch = 0; ch < D; ch++) a[ch] = 0;
            for (int j = 0; j < 3 && tt + j < T; j++) {
                const float *sw = c->swe + j * 3 * D + w * D, *g = DP + ((size_t)w * N + t + j) * D;
                for (int ch = 0; ch < D; ch++) a[ch] += sw[ch] * g[ch];
            }
        }
    }
}

static void forward(const int *tok, int nb, int train) {
    int n = nb * T;
    for (int t = 0; t < n; t++) memcpy(X[0] + t * D, E->w + tok[t] * D, D * sizeof(float));
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; Conv *c = &y->cv;
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_prep(ms[i]);
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {           // phase 1 (per token): rms+quant, Mon_q, Mon_k, Mon_v
            const float *x = X[l] + t * D; int8_t *q = y->q1 + t * D;
            y->r1[t] = rinv(x); y->s1[t] = q8t(x, q, y->r1[t]);
            for (int w = 0; w < 3; w++) mon_fwd(ms[w], q, y->s1[t], y->pre + ((size_t)w * N + t) * D, t);
        }
        mixer_fwd(y, nb, train);
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {           // phase 2 (per token): gate, Mon_o, +res, rms, GLU, +res
            float g[D], o[D], dd;
            float *cc = y->cc + t * D, *q = y->post + (size_t)t * D, *u = VB + t * D, *x = X[l] + t * D, *x1 = y->x1 + t * D;
            float *ga = y->ga + t * D, *ub = y->ub + t * D;
            for (int ch = 0; ch < D; ch++) g[ch] = q[ch] * (cc[ch] + c->be[ch] * u[ch]);
            y->so[t] = q8t(g, y->qo + t * D, 1);
            mon_fwd(&y->o, y->qo + t * D, y->so[t], o, t);
            for (int i = 0; i < D; i++) x1[i] = x[i] + o[i];
            y->r2[t] = rinv(x1); y->s2[t] = q8t(x1, y->q2 + t * D, y->r2[t]);
            mon_fwd(&y->g, y->q2 + t * D, y->s2[t], ga, t);
            mon_fwd(&y->u, y->q2 + t * D, y->s2[t], ub, t);
            #pragma omp simd private(dd)
            for (int i = 0; i < D; i++) g[i] = gelu(ga[i], &dd) * ub[i];
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
        Layer *y = &ly[l]; Conv *c = &y->cv;
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {                   // phase A (per token): GLU, rms2, Mon_o, output gate
            int tid = TID; float a[D], b[D], e[D], dd, *dx = DX + t * D, *ga = y->ga + t * D, *ub = y->ub + t * D;
            mon_bwd(&y->d, y->qd + t * D, y->sd[t], dx, a, t, tid);
            #pragma omp simd private(dd)
            for (int i = 0; i < D; i++) { float gl = gelu(ga[i], &dd); b[i] = a[i] * ub[i] * dd; a[i] *= gl; }
            mon_bwd(&y->u, y->q2 + t * D, y->s2[t], a, e, t, tid);
            mon_bwd(&y->g, y->q2 + t * D, y->s2[t], b, a, t, tid);
            for (int i = 0; i < D; i++) a[i] += e[i];
            rms_bwd(y->x1 + t * D, y->r2[t], a, dx);
            mon_bwd(&y->o, y->qo + t * D, y->so[t], dx, a, t, tid);
            float *cc = y->cc + t * D, *q = y->post + (size_t)t * D, *k = q + (size_t)N * D, *v = k + (size_t)N * D;
            for (int ch = 0; ch < D; ch++) {            // DP[0] <- dq; cc <- d(conv out)
                float u = sc8(k[ch] * v[ch], 1 / c->as[3]) * c->as[3];   // the int8 u the forward used
                DP[(size_t)t * D + ch] = a[ch] * (cc[ch] + c->be[ch] * u); cc[ch] = a[ch] * q[ch];
            }
        }
        mixer_bwd(y, nb);
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {                   // phase B (per token): Mon_q/k/v, rms1
            int tid = TID; float b[D], sum[D] = {0};
            Mon *ms[] = {&y->q, &y->k, &y->v};
            for (int w = 0; w < 3; w++) {
                mon_bwd(ms[w], y->q1 + t * D, y->s1[t], y->pre + ((size_t)w * N + t) * D, b, t, tid);
                for (int ch = 0; ch < D; ch++) sum[ch] += b[ch];
            }
            rms_bwd(X[l] + t * D, y->r1[t], sum, DX + t * D);
        }
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_reduce(ms[i]);
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

// ---- inference: one token at a time, trits x int8 throughout -------------------------------------
// Per layer the state is the last two q/k/v codes (short conv) and a T-long ring of u codes (long conv).
// The long conv is the direct causal sum over the ternary kernel: no FFT, no window recompute, and all-zero
// kernel rows (far taps decay to 0) are skipped. The embedding / head stay fp32.
#if T <= 256
typedef int16_t cacc;                         // |sum of T trit * int8| <= 127 T fits int16
#else
typedef int32_t cacc;
#endif
typedef struct { int8_t hist[2][3][D], ring[T][D]; int hrow[T], nh; } IState;
static IState ist[L];
static long ipos;
static float *ET;                             // embedding transposed [D][VP] for the head: 65 independent sums
#define VP ((V + 15) / 16 * 16)
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__) && M % 16 == 0
#define VNNI 1
// Monarch forward on vpdpbusd (u8 x s8, 4-term dots into int32), 16 lanes per register: trits are stored as
// w + 1 in {0, 1, 2} and the input's sum is subtracted, so the result is exact. R: lane o of block b dots
// x_b[4k..4k+3] (broadcast) with R_b[o][4k..4k+3]. L: lane b of output row o dots q[4k..4k+3][b] with
// L_o[4k..4k+3][b]; vpermb turns four 16-wide tile row pieces into that [b][j] order in one instruction,
// so the permutations still never touch memory.
#include <immintrin.h>
#define MK (M / 4)                                 // 4-term groups per block row
#define MH (M / 16)                                // 16-lane registers per block row
typedef struct { uint8_t r[W3], l[W3]; } IMon;
static IMon imon[L][7];
static void imon_prep(IMon *im, const Mon *m) {
    for (int b = 0; b < M; b++) for (int k = 0; k < MK; k++) for (int o = 0; o < M; o++) for (int j = 0; j < 4; j++)
        im->r[((b * MK + k) * M + o) * 4 + j] = m->rq[(b * M + 4 * k + j) * M + o] + 1;
    for (int o = 0; o < M; o++) for (int k = 0; k < MK; k++) for (int b = 0; b < M; b++) for (int j = 0; j < 4; j++)
        im->l[((o * MK + k) * M + b) * 4 + j] = m->lq[(o * M + 4 * k + j) * M + b] + 1;
}
static void imon_fwd(const IMon *im, const Mon *m, const int8_t *x, float sx, float *y) {
    static const uint8_t pm[64] = {0,16,32,48, 1,17,33,49, 2,18,34,50, 3,19,35,51, 4,20,36,52, 5,21,37,53, 6,22,38,54, 7,23,39,55,
                                   8,24,40,56, 9,25,41,57, 10,26,42,58, 11,27,43,59, 12,28,44,60, 13,29,45,61, 14,30,46,62, 15,31,47,63};
    const __m512i ones = _mm512_set1_epi8(1), perm = _mm512_loadu_si512(pm);
    int32_t z[D]; __m512i mx = _mm512_setzero_si512();
    for (int b = 0; b < M; b++) {
        __m512i acc[MH], sum = _mm512_setzero_si512();
        for (int h = 0; h < MH; h++) acc[h] = _mm512_setzero_si512();
        for (int k = 0; k < MK; k++) {
            int32_t x4; memcpy(&x4, x + b * M + 4 * k, 4); __m512i xv = _mm512_set1_epi32(x4);
            for (int h = 0; h < MH; h++) acc[h] = _mm512_dpbusd_epi32(acc[h], _mm512_loadu_si512(im->r + ((b * MK + k) * M + 16 * h) * 4), xv);
            sum = _mm512_dpbusd_epi32(sum, ones, xv);
        }
        for (int h = 0; h < MH; h++) {
            __m512i zz = _mm512_sub_epi32(acc[h], sum); mx = _mm512_max_epi32(mx, _mm512_abs_epi32(zz));
            _mm512_storeu_si512(z + b * M + 16 * h, zz);
        }
    }
    int m32 = _mm512_reduce_max_epi32(mx); if (m32 < 1) m32 = 1;
    float qs = 127.f / m32, sm = sx * m->rs * m32 / 127;
    __m512 vqs = _mm512_set1_ps(qs);
    int8_t q[D];
    for (int k = 0; k < D; k += 16) _mm_storeu_si128((__m128i *)(q + k), _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_loadu_si512(z + k)), vqs))));
    __m512i qt[MK][MH], qsum[MH];
    for (int h = 0; h < MH; h++) {
        qsum[h] = _mm512_setzero_si512();
        for (int k = 0; k < MK; k++) {                     // rows 4k..4k+3, lanes 16h..16h+15, as [b][j]
            const int8_t *r0 = q + 4 * k * M + 16 * h;
            __m512i rows = _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(_mm_loadu_si128((const __m128i *)r0)),
                            _mm_loadu_si128((const __m128i *)(r0 + M)), 1), _mm_loadu_si128((const __m128i *)(r0 + 2 * M)), 2), _mm_loadu_si128((const __m128i *)(r0 + 3 * M)), 3);
            qt[k][h] = _mm512_permutexvar_epi8(perm, rows); qsum[h] = _mm512_dpbusd_epi32(qsum[h], ones, qt[k][h]);
        }
    }
    __m512 sc = _mm512_set1_ps(sm), ls = _mm512_set1_ps(m->ls);
    for (int o = 0; o < M; o++) for (int h = 0; h < MH; h++) {
        __m512i acc = _mm512_setzero_si512();
        for (int k = 0; k < MK; k++) acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(im->l + ((o * MK + k) * M + 16 * h) * 4), qt[k][h]);
        _mm512_storeu_ps(y + o * M + 16 * h, _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc, qsum[h])), sc), ls));
    }
}
#define IMON(l, i, m, x, sx, y) imon_fwd(&imon[l][i], m, x, sx, y)
#else
#define IMON(l, i, m, x, sx, y) mon_fwd(m, x, sx, y, 0)
#endif
static void infer_prep(void) {                // freeze the trits once
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; Conv *c = &y->cv; IState *st = &ist[l];
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_prep(ms[i]);
#ifdef VNNI
        for (int i = 0; i < 7; i++) imon_prep(&imon[l][i], ms[i]);
#endif
        filter_fwd(c); conv_prep(c);
        st->nh = 0;
        for (int j = 0; j < T; j++) { int nz = 0; for (int ch = 0; ch < D; ch++) nz |= c->hq[j * D + ch]; if (nz) st->hrow[st->nh++] = j; }
    }
    free(ET); ET = fa((size_t)D * VP);
    for (int c = 0; c < V; c++) for (int i = 0; i < D; i++) ET[i * VP + c] = E->w[c * D + i];
}
static void infer_reset(void) { for (int l = 0; l < L; l++) { memset(ist[l].hist, 0, sizeof ist[l].hist); memset(ist[l].ring, 0, sizeof ist[l].ring); } ipos = 0; }
static void infer_step(int tok, float *logits) {
    float x[D], x1[D], o[D], f[D], g[D]; int8_t q[D], cur[3][D]; long p = ipos++;
    memcpy(x, E->w + tok * D, sizeof x);
    for (int l = 0; l < L; l++) {
        Layer *y = &ly[l]; Conv *c = &y->cv; IState *st = &ist[l];
        Mon *ms[] = {&y->q, &y->k, &y->v};
        float s1 = q8t(x, q, rinv(x));
        for (int w = 0; w < 3; w++) { IMON(l, w, ms[w], q, s1, f); float is = 1 / c->as[w]; for (int ch = 0; ch < D; ch++) cur[w][ch] = sc8(f[ch], is); }
        float post[3][D];
        for (int w = 0; w < 3; w++) {         // ternary short conv over the last three codes
            const int8_t *w0 = c->swq + w * D, *w1 = w0 + 3 * D, *w2 = w1 + 3 * D, *h0 = st->hist[0][w], *h1 = st->hist[1][w];
            float sc = c->sws[w] * c->as[w];
            for (int ch = 0; ch < D; ch++) post[w][ch] = (w0[ch] * cur[w][ch] + w1[ch] * h0[ch] + w2[ch] * h1[ch]) * sc + c->sbe[w * D + ch];
        }
        memcpy(st->hist[1], st->hist[0], sizeof st->hist[0]); memcpy(st->hist[0], cur, sizeof cur);
        int8_t *u = st->ring[p % T];
        float iu = 1 / c->as[3]; for (int ch = 0; ch < D; ch++) u[ch] = sc8(post[1][ch] * post[2][ch], iu);
        cacc acc[D] = {0};                    // long conv: sum_j h[j] u[p - j] over the nonzero kernel rows
        for (int r = 0; r < st->nh && st->hrow[r] <= p; r++) {
            int j = st->hrow[r]; const int8_t *hj = c->hq + j * D, *uj = st->ring[(p - j) % T];
            for (int ch = 0; ch < D; ch++) acc[ch] += hj[ch] * uj[ch];
        }
        for (int ch = 0; ch < D; ch++) g[ch] = post[0][ch] * (acc[ch] * c->hs[ch] + c->be[ch] * u[ch]) * c->as[3];
        float so = q8t(g, q, 1); IMON(l, 3, &y->o, q, so, o);
        for (int i = 0; i < D; i++) x1[i] = x[i] + o[i];
        float s2 = q8t(x1, q, rinv(x1)); IMON(l, 4, &y->g, q, s2, f); IMON(l, 5, &y->u, q, s2, g);
        for (int i = 0; i < D; i++) g[i] *= gelu0(f[i]);
        float sd = q8t(g, q, 1); IMON(l, 6, &y->d, q, sd, o);
        for (int i = 0; i < D; i++) x[i] = x1[i] + o[i];
    }
    float r = rinv(x), z[VP];                  // head: all logits at once, over the transposed embedding
    for (int c = 0; c < VP; c++) z[c] = 0;
    for (int i = 0; i < D; i++) { float xi = x[i] * r; const float *e = ET + i * VP; for (int c = 0; c < VP; c++) z[c] += xi * e[c]; }
    memcpy(logits, z, V * sizeof(float));
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
    long nparam = 0; for (int k = 0; k < np; k++) nparam += ps[k].n;
    long trits = (long)L * (7 * 2 * W3 + T * D + 13 * D);   // Monarchs, long kernel, short conv + biases
    printf("moth: vocab %d, d %d, layers %d, T %d, %.2fM params to train, inference on %.2fM trits, %d threads\n",
           V, D, L, T, nparam / 1e6, trits / 1e6, NT);

    int *tok = malloc(N * sizeof(int)), *tgt = malloc(N * sizeof(int));
    double t0 = now();
    for (int step = -1; step <= STEPS; step++) {        // steps -1, 0 calibrate the delayed scales
        seed = hash(step + 2);
        batch(tok, tgt, 0);
        forward(tok, B, 1);
        float loss = xent(tgt, N, 1);
        backward(tok, B);
        adam(step);
        if (step % 50 == 0 && step > 0) { printf("step %4d | loss %.4f | %.0f ms/step\n", step, loss, (now() - t0) * 1e3 / 50); fflush(stdout); t0 = now(); }
        if (step % 500 == 0 && step > 0) {
            float vl = 0; for (int k = 0; k < 8; k++) { batch(tok, tgt, 1); forward(tok, B, 0); vl += xent(tgt, N, 0) / 8; }
            printf("step %4d | val loss %.4f\n", step, vl); t0 = now();
        }
    }

    infer_prep();                                        // check the integer engine against the training forward
    batch(tok, tgt, 1); forward(tok, 1, 0);
    float *lg = malloc(V * sizeof(float)), err = 0, mag = 0; infer_reset();
    for (int t = 0; t < T; t++) {
        infer_step(tok[t], lg);
        for (int c = 0; c < V; c++) { err = fmaxf(err, fabsf(lg[c] - LG[t * V + c])); mag = fmaxf(mag, fabsf(LG[t * V + c])); }
    }
    printf("inference: %ld trits (%.0f KB packed 5 per byte) + fp32 embedding; engine vs training forward: max |dlogit| %.1e (logits up to %.1f)\n",
           trits, trits / 5 / 1024., err, mag);
    int c = data[0], ngen = 2000; char *out = malloc(ngen + 1);   // sample, timed
    infer_reset(); double ts = now();
    for (int k = 0; k < ngen; k++) {
        infer_step(c, lg);
        float mx = -1e30f, s = 0, r;
        for (int j = 0; j < V; j++) mx = fmaxf(mx, lg[j]);
        for (int j = 0; j < V; j++) s += (lg[j] = expf((lg[j] - mx) / 0.8f));
        c = 0; r = urand() * s; while (c < V - 1 && (r -= lg[c]) > 0) c++;
        out[k] = itos[c];
    }
    ts = now() - ts; out[ngen] = 0;
    printf("generated %d tokens in %.3f s: %.0f tokens/s, one thread\n%.600s", ngen, ts, ngen / ts, out);
    putchar('\n');
    return 0;
}
