// moth.c: Moth, a tiny ternary Monarch Mixer byte-level LM in one file of pure C (nanoGPT style). No tokenizer:
// it reads raw bytes and groups them into dynamic patches, as in Meta's Byte Latent Transformer.
//
// BLT (Pagnoni et al. 2024, "Byte Latent Transformer: Patches Scale Better Than Tokens"; reference code at
// github.com/facebookresearch/blt), adapted to moth:
//   x_t = E[byte_t] + sum_{n=3..8} H_n[hash(bytes t-n+1..t)]   hash n-gram embeddings (BLT's rolling polynomial
//                                                                hash over its primes), the model's "vocabulary"
//   e = Encoder(x)             LE layers over bytes
//   g_k = max_{t in patch k} e_t                                 a patch starts where the next-byte entropy of a
//   z = Global(g)              LG layers over patches              small byte LM crosses a global threshold; the
//   d_t = e_t + z_{c(t)}       c(t): latest patch complete         first patch is one byte; patches cap at PMAX
//   logits_t = Head(Decoder(d)) LD layers over bytes              (BLT's "lag fix": byte t reads the global state
//                                                                  of the last patch that ends at or before t)
//   The entropy model is itself a small moth (LH layers), trained first; its entropies are computed once.
//   At inference the global stack runs once per patch, not per byte.
//
// Every stack is the same all-ternary M2 block (Monarch Mixer, Fu et al. 2023, causal / GPT form):
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
//   ./moth input.txt -o run.ck         also save a checkpoint every CKEVERY steps (resume with -r run.ck)
//   ./moth -g run.ck -p "prompt" -n 1000   no training: load a checkpoint and continue the prompt
//   cc ... -DLCTX=4096 -DB=1 moth.c        BLT's long global context: 4096-byte training sequences, the byte stacks
//                                          in TB-byte chunks, the global stack over the whole sequence's patches
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
#ifndef TB
#define TB 128          // bytes per window: the byte stacks' conv length (power of 2)
#endif
#ifndef LCTX
#define LCTX TB         // bytes per training sequence. LCTX > TB is BLT's long global context: the byte
#endif                  //   stacks run in TB-byte chunks, the global stack over the whole sequence's patches
#define NCK (LCTX / TB) // byte-stack chunks per sequence
#define TG (LCTX / 2)   // patch slots per sequence: the global stack's conv length
#ifndef LE
#define LE 1            // local encoder layers
#endif
#ifndef LG
#define LG 4            // global layers
#endif
#ifndef LD
#define LD 2            // local decoder layers
#endif
#ifndef LH
#define LH 1            // entropy model layers
#endif
#ifndef B
#define B 16            // sequences per batch (B * LCTX bytes a step)
#endif
#define NW (B * NCK)    // byte-stack sequences (chunks) per batch
#define V 256           // bytes
#define NG 6            // hash n-gram embeddings, n = 3 .. 8
#ifndef HV
#define HV 256          // rows per hash table (2048 = a BERT-tiny-sized 3.9M model; overfits 1 MB of text)
#endif
#ifndef PSZ
#define PSZ 4.0         // target mean patch size, in bytes: sets the entropy threshold
#endif
#define PMAX 16         // longest patch
#ifndef STEPS
#define STEPS 3000
#endif
#ifndef HSTEPS
#define HSTEPS 1000     // entropy model steps
#endif
#define LR 3e-3f
#if defined(__AVX512VNNI__) && defined(__AVX512VBMI__) && M % 16 == 0
#define VNNI 1                                 // int8 dot-product kernels (vpdpbusd), else portable C
#include <immintrin.h>
static const uint8_t PM4[64] = {0,16,32,48, 1,17,33,49, 2,18,34,50, 3,19,35,51, 4,20,36,52, 5,21,37,53, 6,22,38,54, 7,23,39,55,
                                8,24,40,56, 9,25,41,57, 10,26,42,58, 11,27,43,59, 12,28,44,60, 13,29,45,61, 14,30,46,62, 15,31,47,63};
// four 16-byte rows -> one register, lane l holding byte l of each row: [l][row], the 4-term groups vpdpbusd wants
#define ROWS4(p0, p1, p2, p3) _mm512_permutexvar_epi8(_mm512_loadu_si512(PM4), _mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4( \
    _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *)(p0))), _mm_loadu_si128((const __m128i *)(p1)), 1), \
    _mm_loadu_si128((const __m128i *)(p2)), 2), _mm_loadu_si128((const __m128i *)(p3)), 3))
#endif
#define MAXF(a, b) ((a) > (b) ? (a) : (b))   // unlike fmaxf, vectorises without -ffast-math

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static float urand(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (rs >> 40) / 16777216.f; }
static float randn(void) { return sqrtf(-2 * logf(urand() + 1e-9f)) * cosf(6.2831853f * urand()); }
static uint32_t hash(uint32_t x) { x ^= x >> 16; x *= 0x7feb352d; x ^= x >> 15; x *= 0x846ca68b; return x ^ (x >> 16); }
static float *fa(size_t n) { return calloc(n, sizeof(float)); }
static int8_t *ia(size_t n) { return calloc(n, 1); }
static int NT;          // threads
static uint32_t seed;   // per-step stochastic rounding seed

typedef struct { float *w, *g, *m, *v; int n; } P;   // master weight, grad, adam moments
static P ps[8 + NG + 22 * (LE + LG + LD + LH)]; static int np;
static P *param(int n, float sd) {
    P *p = &ps[np++]; p->n = n; p->w = fa(n); p->g = fa(n); p->m = fa(n); p->v = fa(n);
    for (int i = 0; i < n; i++) p->w[i] = sd * randn();
    return p;
}

// ---- quantizers ------------------------------------------------------------------------------
static float tern(const float *w, int8_t *q, int n, int st) {        // absmean ternary, returns scale
    float s = 1e-8f; for (int i = 0; i < n; i++) s += fabsf(w[i * st]); s /= n;
    float h = 0.5f * s;                                                // round(w / s) clipped to +-1, without roundf
    for (int i = 0; i < n; i++) q[i * st] = (w[i * st] >= h) - (w[i * st] <= -h);
    return s;
}
static float q8t(const float *x, int8_t *q, float mul) {              // per-token absmax int8; scale*mul
    float m = 1e-8f;
    #pragma omp simd reduction(max:m)
    for (int i = 0; i < D; i++) { float a = fabsf(x[i]); m = a > m ? a : m; }
    float is = 127 / m; for (int i = 0; i < D; i++) q[i] = (int8_t)rintf(x[i] * is);
    return m * mul / 127;
}
static inline float floor_(float x) { float t = (float)(int32_t)x; return t > x ? t - 1 : t; }   // |x| < 2^31; unlike floorf, vectorises
static int8_t sr8(float v, uint32_t idx) {                            // stochastic round + saturate
    v = v > 130.f ? 130.f : v < -130.f ? -130.f : v;                   // (saturates to the same codes; keeps floor_ in range)
    float r = floor_(v + (hash(idx ^ seed) >> 8) * 0x1p-24f);
    return r > 127 ? 127 : r < -127 ? -127 : (int8_t)r;
}
typedef struct { float prev, cur[64][16]; } Amax;                      // delayed gradient scale (a cache line per thread)
static float gscale(Amax *a) { return (a->prev > 0 ? a->prev : 1e-3f) / 127; }
static void roll(Amax *a) { a->prev = 0; for (int t = 0; t < NT; t++) { a->prev = fmaxf(a->prev, a->cur[t][0]); a->cur[t][0] = 0; } }

// ---- Monarch: R (row blocks, weights [b][i][o]) then L (column blocks, weights [o][i][lane]) ----
typedef struct { P *r, *l; int8_t rq[W3], rt[W3], lq[W3], *mq; float rs, ls, *ms; Amax gl, gr; int32_t *dw; uint32_t salt; } Mon;
#define SLAB (2 * W3 + 2 * D)                  // per-thread grad slab: dR [b][o][i], dL [o][i][b], and the two
                                               // offset corrections the 4-token VNNI path needs (cR [b][o], cL [o][b])
static void mon_init(Mon *m, float sd, uint32_t salt, int ntok) {
    m->r = param(W3, sd / sqrtf(M)); m->l = param(W3, 1 / sqrtf(M));
    m->mq = ia((size_t)ntok * D); m->ms = fa(ntok); m->dw = calloc((size_t)NT * SLAB, 4); m->salt = salt * 0x9E3779B9u;
}
static void mon_prep(Mon *m) {                                              // once per step, O(params):
    m->rs = tern(m->r->w, m->rq, W3, 1); m->ls = tern(m->l->w, m->lq, W3, 1);  // a 4KB ternary transpose so R^T
    for (int b = 0; b < M; b++) for (int i = 0; i < M; i++) for (int o = 0; o < M; o++)  // is broadcast-accumulate too
        m->rt[(b * M + o) * M + i] = m->rq[(b * M + i) * M + o];
}
// token t: x = int8 codes with scale sx -> y (float). Stores the int8 mid codes for the backward.
static void __attribute__((unused)) mon_fwd(Mon *m, const int8_t *x, float sx, float *y, int t) {
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
// token t: dy (float) -> dx (float), and the int8 grad codes at L's output (g1) and R's output (g2), which
// mon_dw4 turns into weight grads four tokens at a time
static void __attribute__((unused)) mon_bwdx(Mon *m, float sx, const float *dy, float *dx, int t, int tid, int8_t *g1, int8_t *g2) {
    float sm = m->ms[t], gl = gscale(&m->gl), gr = gscale(&m->gr), am = 0;
    int16_t dm[D] = {0};
    #pragma omp simd reduction(max:am)
    for (int k = 0; k < D; k++) { float v = dy[k] * sm, a = fabsf(v); am = a > am ? a : am; g1[k] = sr8(v / gl, (t * D + k) ^ m->salt); }
    m->gl.cur[tid][0] = MAXF(m->gl.cur[tid][0], am);
    for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) {                 // L^T g, lane-wise
        const int8_t *w = m->lq + (o * M + i) * M, *go = g1 + o * M; int16_t *a = dm + i * M;
        for (int b = 0; b < M; b++) a[b] += w[b] * go[b];
    }
    float c = gl * m->ls / sm * sx / gr; am = 0;                             // fold sx so dR reuses x's codes
    #pragma omp simd reduction(max:am)
    for (int k = 0; k < D; k++) { float v = dm[k] * c, a = fabsf(v); am = a > am ? a : am; g2[k] = sr8(v, (t * D + k) ^ ~m->salt); }
    m->gr.cur[tid][0] = MAXF(m->gr.cur[tid][0], am * gr);
    float cx = gr * m->rs / sx; int16_t ax[D] = {0};
    for (int b = 0; b < M; b++) for (int o = 0; o < M; o++) {                 // R^T g, broadcast g_bo
        int go = g2[b * M + o]; const int8_t *w = m->rt + (b * M + o) * M; int16_t *a = ax + b * M;
        for (int i = 0; i < M; i++) a[i] += go * w[i];
    }
    for (int k = 0; k < D; k++) dx[k] = ax[k] * cx;
}
// weight grads of tokens t0..t0+3 into thread tid's slab: dL[o][i][b] += sum_j g1_j[o][b] q_j[i][b] (q: the
// token's mid codes), dR[b][o][i] += sum_j x_j[b][i] g2_j[b][o]. Padding tokens carry zero codes.
static void mon_dw4(Mon *m, int tid, int t0, const int8_t *xbase, int8_t (*g1)[D], int8_t (*g2)[D]) {
    int32_t *dwr = m->dw + (size_t)tid * SLAB, *dwl = dwr + W3;
    const int8_t *q[4], *x[4];
    for (int j = 0; j < 4; j++) { q[j] = m->mq + (size_t)(t0 + j) * D; x[j] = xbase + (size_t)(t0 + j) * D; }
#ifdef VNNI
    // vpdpbusd wants one unsigned operand: use (code + 128) = code ^ 0x80, and count sum_j g per output in
    // cR / cL so mon_reduce can take the 128 sum_j g back out. vpermb interleaves the four tokens per lane.
    const __m512i flip = _mm512_set1_epi8((char)0x80), ones = _mm512_set1_epi8(1);
    int32_t *cR = dwr + 2 * W3, *cL = cR + D;
    for (int h = 0; h < M / 16; h++) {                                        // dL: lanes b = 16h..16h+15
        __m512i gt[M], qt[M];
        for (int o = 0; o < M; o++) {
            gt[o] = ROWS4(g1[0] + o * M + 16 * h, g1[1] + o * M + 16 * h, g1[2] + o * M + 16 * h, g1[3] + o * M + 16 * h);
            __m512i c = _mm512_loadu_si512(cL + o * M + 16 * h);
            _mm512_storeu_si512(cL + o * M + 16 * h, _mm512_dpbusd_epi32(c, ones, gt[o]));
        }
        for (int i = 0; i < M; i++) qt[i] = _mm512_xor_si512(ROWS4(q[0] + i * M + 16 * h, q[1] + i * M + 16 * h, q[2] + i * M + 16 * h, q[3] + i * M + 16 * h), flip);
        for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) {
            int32_t *p = dwl + (o * M + i) * M + 16 * h;
            _mm512_storeu_si512(p, _mm512_dpbusd_epi32(_mm512_loadu_si512(p), qt[i], gt[o]));
        }
        for (int b = 0; b < M; b++) {                                         // dR: lanes i = 16h..16h+15
            __m512i xt = _mm512_xor_si512(ROWS4(x[0] + b * M + 16 * h, x[1] + b * M + 16 * h, x[2] + b * M + 16 * h, x[3] + b * M + 16 * h), flip);
            for (int o = 0; o < M; o++) {
                int k = b * M + o; uint32_t g4 = (uint8_t)g2[0][k] | (uint8_t)g2[1][k] << 8 | (uint8_t)g2[2][k] << 16 | (uint32_t)(uint8_t)g2[3][k] << 24;
                if (h == 0) cR[k] += g2[0][k] + g2[1][k] + g2[2][k] + g2[3][k];
                int32_t *p = dwr + k * M + 16 * h;
                _mm512_storeu_si512(p, _mm512_dpbusd_epi32(_mm512_loadu_si512(p), xt, _mm512_set1_epi32((int32_t)g4)));
            }
        }
    }
#else
    for (int j = 0; j < 4; j++) {
        for (int o = 0; o < M; o++) for (int i = 0; i < M; i++) {
            int32_t *dl = dwl + (o * M + i) * M; const int8_t *go = g1[j] + o * M, *qi = q[j] + i * M;
            for (int b = 0; b < M; b++) dl[b] += go[b] * qi[b];
        }
        for (int b = 0; b < M; b++) for (int o = 0; o < M; o++) {
            int go = g2[j][b * M + o]; int32_t *dr = dwr + (b * M + o) * M; const int8_t *xb = x[j] + b * M;
            for (int i = 0; i < M; i++) dr[i] += xb[i] * go;
        }
    }
#endif
}
static void mon_reduce(Mon *m) {                                             // sum thread slabs, O(params)
    float gr = gscale(&m->gr), gl = gscale(&m->gl);
    #pragma omp parallel for
    for (int k = 0; k < W3; k++) {
        int32_t sr = 0, sl = 0;
        int b = k / (M * M), o = k / M % M, i = k % M;             // (for dL: o = b, i = o, b = i)
        for (int t = 0; t < NT; t++) {
            int32_t *s = m->dw + (size_t)t * SLAB;
            sr += s[k] - 128 * s[2 * W3 + k / M]; sl += s[W3 + k] - 128 * s[2 * W3 + D + b * M + i]; s[k] = s[W3 + k] = 0;
        }
        m->r->g[(b * M + i) * M + o] += sr * gr; m->l->g[k] += sl * gl;
    }
    for (int t = 0; t < NT; t++) memset(m->dw + (size_t)t * SLAB + 2 * W3, 0, 2 * D * sizeof(int32_t));
    roll(&m->gr); roll(&m->gl);
}

#ifdef VNNI
// Monarch forward on vpdpbusd (u8 x s8, 4-term dots into int32), 16 lanes per register: trits are stored as
// w + 1 in {0, 1, 2} and the input's sum is subtracted, so the result is exact. R: lane o of block b dots
// x_b[4k..4k+3] (broadcast) with R_b[o][4k..4k+3]. L: lane b of output row o dots q[4k..4k+3][b] with
// L_o[4k..4k+3][b]; vpermb turns four 16-wide tile row pieces into that [b][j] order in one instruction,
// so the permutations still never touch memory.
#define MK (M / 4)                                 // 4-term groups per block row
#define MH (M / 16)                                // 16-lane registers per block row
typedef struct { uint8_t r[W3], l[W3], lt[W3], rt[W3]; } IMon;   // forward R, L; backward L^T, R^T
static void imon_prep(IMon *im, const Mon *m) {
    for (int b = 0; b < M; b++) for (int k = 0; k < MK; k++) for (int o = 0; o < M; o++) for (int j = 0; j < 4; j++)
        im->r[((b * MK + k) * M + o) * 4 + j] = m->rq[(b * M + 4 * k + j) * M + o] + 1;
    for (int o = 0; o < M; o++) for (int k = 0; k < MK; k++) for (int b = 0; b < M; b++) for (int j = 0; j < 4; j++)
        im->l[((o * MK + k) * M + b) * 4 + j] = m->lq[(o * M + 4 * k + j) * M + b] + 1;
    for (int i = 0; i < M; i++) for (int k = 0; k < MK; k++) for (int b = 0; b < M; b++) for (int j = 0; j < 4; j++)
        im->lt[((i * MK + k) * M + b) * 4 + j] = m->lq[((4 * k + j) * M + i) * M + b] + 1;   // L^T: sum over o
    for (int b = 0; b < M; b++) for (int k = 0; k < MK; k++) for (int i = 0; i < M; i++) for (int j = 0; j < 4; j++)
        im->rt[((b * MK + k) * M + i) * 4 + j] = m->rq[(b * M + i) * M + 4 * k + j] + 1;     // R^T: sum over o
}
static void imon_fwd(const IMon *im, const Mon *m, const int8_t *x, float sx, float *y, int8_t *mid, float *msc) {
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
    int8_t qb[D], *q = mid ? mid : qb; if (msc) *msc = sm;
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
// mon_bwdx on vpdpbusd: same arithmetic (exact), the two matvecs as 4-term int8 dots
static void imon_bwdx(const IMon *im, Mon *m, float sx, const float *dy, float *dx, int t, int tid, int8_t *g1, int8_t *g2) {
    float sm = m->ms[t], gl = gscale(&m->gl), gr = gscale(&m->gr), am = 0;
    const __m512i ones = _mm512_set1_epi8(1);
    #pragma omp simd reduction(max:am)
    for (int k = 0; k < D; k++) { float v = dy[k] * sm, a = fabsf(v); am = a > am ? a : am; g1[k] = sr8(v / gl, (t * D + k) ^ m->salt); }
    m->gl.cur[tid][0] = MAXF(m->gl.cur[tid][0], am);
    int32_t dm[D], ax[D];
    for (int h = 0; h < MH; h++) {                                            // dm[i][b] = sum_o L[o][i][b] g1[o][b]
        __m512i gt[MK], gs = _mm512_setzero_si512();
        for (int k = 0; k < MK; k++) {
            const int8_t *p = g1 + 4 * k * M + 16 * h;
            gt[k] = ROWS4(p, p + M, p + 2 * M, p + 3 * M); gs = _mm512_dpbusd_epi32(gs, ones, gt[k]);
        }
        for (int i = 0; i < M; i++) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < MK; k++) acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(im->lt + ((i * MK + k) * M + 16 * h) * 4), gt[k]);
            _mm512_storeu_si512(dm + i * M + 16 * h, _mm512_sub_epi32(acc, gs));
        }
    }
    float c = gl * m->ls / sm * sx / gr; am = 0;
    #pragma omp simd reduction(max:am)
    for (int k = 0; k < D; k++) { float v = dm[k] * c, a = fabsf(v); am = a > am ? a : am; g2[k] = sr8(v, (t * D + k) ^ ~m->salt); }
    m->gr.cur[tid][0] = MAXF(m->gr.cur[tid][0], am * gr);
    for (int b = 0; b < M; b++) {                                             // ax[b][i] = sum_o R[b][i][o] g2[b][o]
        int gsum = 0; for (int o = 0; o < M; o++) gsum += g2[b * M + o];
        for (int h = 0; h < MH; h++) {
            __m512i acc = _mm512_setzero_si512();
            for (int k = 0; k < MK; k++) {
                int32_t g4; memcpy(&g4, g2 + b * M + 4 * k, 4);
                acc = _mm512_dpbusd_epi32(acc, _mm512_loadu_si512(im->rt + ((b * MK + k) * M + 16 * h) * 4), _mm512_set1_epi32(g4));
            }
            _mm512_storeu_si512(ax + b * M + 16 * h, _mm512_sub_epi32(acc, _mm512_set1_epi32(gsum)));
        }
    }
    float cx = gr * m->rs / sx;
    for (int k = 0; k < D; k++) dx[k] = ax[k] * cx;
}
#define BWDX(y, i, m, sx, dy, dx, t, tid, g1, g2) imon_bwdx(&(y)->im[i], m, sx, dy, dx, t, tid, g1, g2)
#define IMON(y, i, m, x, sx, out) imon_fwd(&(y)->im[i], m, x, sx, out, NULL, NULL)
#define TMON(y, i, m, x, sx, out, t) imon_fwd(&(y)->im[i], m, x, sx, out, (m)->mq + (size_t)(t) * D, (m)->ms + (t))   // training: keeps mids
#else
#define IMON(y, i, m, x, sx, out) mon_fwd(m, x, sx, out, 0)
#define BWDX(y, i, m, sx, dy, dx, t, tid, g1, g2) mon_bwdx(m, sx, dy, dx, t, tid, g1, g2)
#define TMON(y, i, m, x, sx, out, t) mon_fwd(m, x, sx, out, t)
#endif

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
#define CH 16                                 // channels per vector lane group
#define NCH (D / CH)
typedef struct { int T, NF, P1, P2; float *F1r, *F1i, *F2r, *F2i, *TWr, *TWi; } Plan;   // one per sequence length
static float *SCR;                                        // per-thread FFT scratch, sized for the longest stack
static void plan_init(Plan *p, int T) {
    int NF = 2 * T, lg = 0; while ((1 << lg) < NF) lg++;
    int P1 = 1 << lg / 2, P2 = NF / P1;
    p->T = T; p->NF = NF; p->P1 = P1; p->P2 = P2;
    p->F1r = fa(P1 * P1); p->F1i = fa(P1 * P1); p->F2r = fa(P2 * P2); p->F2i = fa(P2 * P2); p->TWr = fa(NF); p->TWi = fa(NF);
    for (int k = 0; k < P1 * P1; k++) { p->F1r[k] = cos(2 * M_PI * (k / P1) * (k % P1) / P1); p->F1i[k] = -sin(2 * M_PI * (k / P1) * (k % P1) / P1); }
    for (int k = 0; k < P2 * P2; k++) { p->F2r[k] = cos(2 * M_PI * (k / P2) * (k % P2) / P2); p->F2i[k] = -sin(2 * M_PI * (k / P2) * (k % P2) / P2); }
    for (int k = 0; k < NF; k++) { p->TWr[k] = cos(2 * M_PI * (k / P2) * (k % P2) / NF); p->TWi[k] = -sin(2 * M_PI * (k / P2) * (k % P2) / NF); }
}
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
static void twid(const Plan *p, float *re, float *im, float s, float sg) {   // [k1][n2][CH] *= s TW^sg, sg = +-1
    for (int k = 0; k < p->NF; k++) {
        float c = p->TWr[k] * s, d = sg * p->TWi[k] * s, *a = re + k * CH, *b = im + k * CH;
        for (int l = 0; l < CH; l++) { float r = a[l] * c - b[l] * d; b[l] = a[l] * d + b[l] * c; a[l] = r; }
    }
}
// x (+ i y), T steps of CH lanes at stride xs (y may be NULL) -> spectrum (or, oi) [P1][P2][CH]
static void mfft(const Plan *p, const float *x, const float *y, int xs, float *or, float *oi, float *tr, float *ti) {
    int T = p->T, P1 = p->P1, P2 = p->P2, R = P2 * CH; float *rr = tr + T * CH, *ri = ti + T * CH;
    for (int t = 0; t < T; t++) for (int l = 0; l < CH; l++) { tr[t * CH + l] = x[t * xs + l]; ti[t * CH + l] = y ? y[t * xs + l] : 0; }
    bdft(tr, ti, or, oi, P1, P1 / 2, P1, p->F1r, p->F1i, R, 1);  // columns: P1 x P1/2 (zero half skipped)
    twid(p, or, oi, 1, 1);
    for (int k1 = 0; k1 < P1; k1++) {                              // rows: P2 x P2
        memcpy(rr, or + k1 * R, R * 4); memcpy(ri, oi + k1 * R, R * 4);
        bdft(rr, ri, or + k1 * R, oi + k1 * R, P2, P2, P2, p->F2r, p->F2i, CH, 1);
    }
}
// spectrum (zr, zi) (destroyed) -> first T steps: real part to x, imaginary part to y (if not NULL), times s
static void mifft(const Plan *p, float *zr, float *zi, float *x, float *y, int xs, float s, float *tr, float *ti) {
    int T = p->T, P1 = p->P1, P2 = p->P2, R = P2 * CH;
    for (int k1 = 0; k1 < P1; k1++) {                              // rows: F_P2^-1
        memcpy(tr, zr + k1 * R, R * 4); memcpy(ti, zi + k1 * R, R * 4);
        bdft(tr, ti, zr + k1 * R, zi + k1 * R, P2, P2, P2, p->F2r, p->F2i, CH, -1);
    }
    twid(p, zr, zi, s, -1);
    bdft(zr, zi, tr, ti, P1 / 2, P1, P1, p->F1r, p->F1i, R, -1);  // columns: only the P1/2 causal outputs
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
    int8_t *hq, swq[9 * D], sbq[3 * D], bq[D];            // trits: long kernel [T][D], short conv, its bias, skip bias
    float hs[D], sws[3], sbs[3], bs;                      // their absmean scales (kernel: per channel; rest: per stream)
    float *he, swe[9 * D], sbe[3 * D], be[D];             // trit * scale, what training runs on
    float amax[4], as[4];                                 // static int8 scales for pre q, k, v and u: EMA amax, amax / 127
} Conv;
typedef struct {
    Mon q, k, v, o, g, u, d; Conv cv;
    int8_t *q1, *qo, *q2, *qd;                // int8 codes: Monarch inputs
    float *s1, *so, *s2, *sd, *r1, *r2;       // per-token scales and rms
    float *pre, *post, *cc, *x1, *ga, *ub;    // float activations kept for backward (pre/post: [3][N][D])
    int8_t hist[2][3][D], *ring;              // inference: last two q/k/v codes, a T-long ring of u codes,
    int *hrow, nh;                            //   and the nonzero rows of the long kernel
#ifdef VNNI
    IMon im[7];
    uint8_t *uh;                              // long conv history for vpdpbusd: u + 128 at [2T positions / 4][D][4]
    uint8_t *kp; int *kg, ko[4][D / 16 + 1];  // packed kernel: nonzero 16-channel x 4-tap blocks, 2 bits a trit (16 B),
    int32_t hc[D];                            //   their groups, per-(phase, chunk) offsets; 128 sum_j h[j] per channel
#endif
} Layer;
// a stack of layers over sequences of length T (bytes or patches); X[0] is its input, X[L] its output
typedef struct { int T, L, N; Plan pl; Layer *ly; float **X; long ipos; const int *len; } Stack;   // len: valid steps per sequence, or NULL
static float *VB, *DP;                        // shared scratch: conv input (grad), short conv out grads

static inline float tanh_(float u) {           // tanh as a clamped [7/6] Pade
    float v = u < -4.97f ? -4.97f : u > 4.97f ? 4.97f : u, v2 = v * v;     // (fminf / fmaxf would block vectorising)
    return v * (135135.f + v2 * (17325.f + v2 * (378.f + v2))) / (135135.f + v2 * (62370.f + v2 * (3150.f + v2 * 28.f)));
}
static inline float gelu0(float x) { return 0.5f * x * (1 + tanh_(0.7978846f * (x + 0.044715f * x * x * x))); }
static float gelu(float x, float *d) {        // tanh GELU and its derivative
    float u = 0.7978846f * (x + 0.044715f * x * x * x), th = tanh_(u);
    *d = 0.5f * (1 + th) + 0.5f * x * (1 - th * th) * 0.7978846f * (1 + 3 * 0.044715f * x * x);
    return 0.5f * x * (1 + th);
}

static void stack_build(Stack *S, int T, int nl, uint32_t salt, int nseq) {   // nseq sequences of T steps
    int N = nseq * T, NF = 2 * T;
    S->T = T; S->L = nl; S->N = N; plan_init(&S->pl, T);
    S->ly = calloc(nl, sizeof(Layer)); S->X = malloc((nl + 1) * sizeof(float *));
    for (int l = 0; l <= nl; l++) S->X[l] = fa((size_t)N * D);
    for (int l = 0; l < nl; l++) {
        Layer *y = &S->ly[l]; Conv *c = &y->cv; float ro = 1 / sqrtf(2 * nl);
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_init(ms[i], ms[i] == &y->o || ms[i] == &y->d ? ro : 1, salt + 10 * l + i + 1, N);
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
        c->hq = ia((size_t)T * D); c->he = fa((size_t)T * D); c->Kr = fa(NF * D); c->Ki = fa(NF * D); c->Zr = fa((size_t)(nseq + 1) / 2 * NF * D); c->Zi = fa((size_t)(nseq + 1) / 2 * NF * D);
        int8_t **q[] = {&y->q1, &y->qo, &y->q2, &y->qd};
        for (int i = 0; i < 4; i++) *q[i] = ia(N * D);
        float **sc[] = {&y->s1, &y->so, &y->s2, &y->sd, &y->r1, &y->r2};
        for (int i = 0; i < 6; i++) *sc[i] = fa(N);
        y->pre = fa(3 * N * D); y->post = fa(3 * N * D); y->cc = fa(N * D); y->x1 = fa(N * D); y->ga = fa(N * D); y->ub = fa(N * D);
        y->ring = ia((size_t)T * D); y->hrow = malloc(T * sizeof(int));
    }
}

static void conv_prep(Conv *c) {                      // ternarise short conv (per stream) and both biases
    for (int w = 0; w < 3; w++) {
        float s = 1e-8f; for (int j = 0; j < 3; j++) for (int ch = 0; ch < D; ch++) s += fabsf(c->sw->w[j * 3 * D + w * D + ch]);
        s /= 3 * D; c->sws[w] = s;
        for (int j = 0; j < 3; j++) for (int ch = 0; ch < D; ch++) {
            int k = j * 3 * D + w * D + ch;
            c->swq[k] = (c->sw->w[k] >= 0.5f * s) - (c->sw->w[k] <= -0.5f * s); c->swe[k] = c->swq[k] * s;
        }
        c->sbs[w] = tern(c->sb->w + w * D, c->sbq + w * D, D, 1);
        for (int ch = 0; ch < D; ch++) c->sbe[w * D + ch] = c->sbq[w * D + ch] * c->sbs[w];
    }
    c->bs = tern(c->bias->w, c->bq, D, 1);
    for (int ch = 0; ch < D; ch++) c->be[ch] = c->bq[ch] * c->bs;
    for (int i = 0; i < 4; i++) c->as[i] = (c->amax[i] > 0 ? c->amax[i] : 1) / 127;
}
static inline int8_t sc8(float x, float is) {   // round(x / scale), saturated; is = 1 / scale. Branch-free: vectorises
    float r = rintf(x * is); r = r > 127.f ? 127.f : r < -127.f ? -127.f : r; return (int8_t)(int32_t)r;
}
// static-scale int8 fake quant (STE): x <- s clip(round(x / s)); returns max |x| for the EMA
static float sq8(float *x, int n, float s) {  // (= sc8(x, 1 / s) * s, written branch-free so it vectorises)
    float m = 0, is = 1 / s;
    #pragma omp simd reduction(max:m)
    for (int i = 0; i < n; i++) {
        float a = fabsf(x[i]), r = rintf(x[i] * is); m = a > m ? a : m;
        r = r > 127.f ? 127.f : r < -127.f ? -127.f : r; x[i] = r * s;
    }
    return m;
}
static inline float fsin(float x) {           // |err| ~1e-7 for moderate |x|: reduce to [-pi/2, pi/2], odd Taylor to x^11
    float r = x - floor_(x * 0.15915494f + 0.5f) * 6.2831853f;
    r = r > 1.5707963f ? 3.1415927f - r : r < -1.5707963f ? -3.1415927f - r : r;
    float r2 = r * r;
    return r * (1 + r2 * (-1 / 6.f + r2 * (1 / 120.f + r2 * (-1 / 5040.f + r2 * (1 / 362880.f - r2 / 39916800.f)))));
}
static inline float fcos(float x) { return fsin(x + 1.5707963f); }
static void filter_fwd(Stack *S, Conv *c) {           // h = FFN(pos) * mod, ternarised per channel, and its spectrum
    int T = S->T, NF = S->pl.NF;
    static float w3t[FO * D];                         // w3 transposed: the last layer accumulates whole rows
    for (int ch = 0; ch < D; ch++) for (int o = 0; o < FO; o++) w3t[o * D + ch] = c->w3->w[ch * FO + o];
    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        float *p1 = c->p1 + t * FO, *a1 = c->a1 + t * FO, *p2 = c->p2 + t * FO, *a2 = c->a2 + t * FO, hr[D] = {0};
        for (int o = 0; o < FO; o++) {
            float s = c->b1->w[o]; for (int e = 0; e < FE; e++) s += c->w1->w[o * FE + e] * c->pz[t * FE + e];
            p1[o] = s;
        }
        for (int o = 0; o < FO; o++) a1[o] = fsin(p1[o]);
        for (int o = 0; o < FO; o++) {
            float s = c->b2->w[o]; for (int i = 0; i < FO; i++) s += c->w2->w[o * FO + i] * a1[i];
            p2[o] = s;
        }
        for (int o = 0; o < FO; o++) a2[o] = fsin(p2[o]);
        for (int o = 0; o < FO; o++) { float a = a2[o]; const float *w = w3t + o * D; for (int ch = 0; ch < D; ch++) hr[ch] += a * w[ch]; }
        for (int ch = 0; ch < D; ch++) c->h[t * D + ch] = hr[ch] * c->mod[t * D + ch];
    }
    for (int ch = 0; ch < D; ch++) c->hs[ch] = 1e-8f; // the kernel inference sees: trits, one absmean scale per channel
    for (int t = 0; t < T; t++) for (int ch = 0; ch < D; ch++) c->hs[ch] += fabsf(c->h[t * D + ch]);   // row-major: no
    for (int ch = 0; ch < D; ch++) c->hs[ch] /= T;                                                    // 1 KB strides
    for (int t = 0; t < T; t++) for (int ch = 0; ch < D; ch++) {
        float x = c->h[t * D + ch], h = 0.5f * c->hs[ch]; int8_t q = (x >= h) - (x <= -h);
        c->hq[t * D + ch] = q; c->he[t * D + ch] = q * c->hs[ch];
    }
    #pragma omp parallel for
    for (int ch = 0; ch < NCH; ch++) {
        float *s = SCR + (size_t)TID * 6 * NF * CH;
        mfft(&S->pl, c->he + ch * CH, NULL, D, c->Kr + ch * NF * CH, c->Ki + ch * NF * CH, s, s + NF * CH);
    }
}
static void filter_bwd(Stack *S, Conv *c) {           // dh -> filter FFN grads
    int T = S->T;
    float *g2 = fa(T * FO), *g1 = fa(T * FO);
    #pragma omp parallel for
    for (int t = 0; t < T; t++) {
        float s[FO] = {0};
        for (int ch = 0; ch < D; ch++) {
            float g = c->dh[t * D + ch] *= c->mod[t * D + ch]; const float *w = c->w3->w + ch * FO;
            for (int o = 0; o < FO; o++) s[o] += g * w[o];
        }
        for (int o = 0; o < FO; o++) g2[t * FO + o] = s[o] * fcos(c->p2[t * FO + o]);
        for (int i = 0; i < FO; i++) {
            float s = 0; for (int o = 0; o < FO; o++) s += g2[t * FO + o] * c->w2->w[o * FO + i];
            g1[t * FO + i] = s * fcos(c->p1[t * FO + i]);
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
static void mixer_fwd(Stack *S, Layer *y, int nb, int train) {
    int T = S->T, N = S->N, NF = S->pl.NF;
    Conv *c = &y->cv; int n = nb * T, np2 = (nb + 1) / 2; float m0 = 0, m1 = 0, m2 = 0, m3 = 0;
    filter_fwd(S, c); conv_prep(c);
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
#ifdef DIRECTCONV                           // tests only: a direct causal sum, so no output depends on a later input
    #pragma omp parallel for                    // even through FFT rounding (forward only; no spectra for backward)
    for (int t = 0; t < n; t++) {
        float *o = y->cc + (size_t)t * D; memset(o, 0, D * sizeof(float));
        for (int j = 0; j <= t % T; j++) { const float *h = c->he + (size_t)j * D, *u = VB + (size_t)(t - j) * D; for (int ch = 0; ch < D; ch++) o[ch] += h[ch] * u[ch]; }
    }
    return;
#endif
    #pragma omp parallel for collapse(2)
    for (int p = 0; p < np2; p++) for (int ch = 0; ch < NCH; ch++) {   // long conv, two sequences at a time
        float *s = SCR + (size_t)TID * 6 * NF * CH, *ur = s + 2 * NF * CH, *ui = ur + NF * CH;
        float *zr = c->Zr + (size_t)(p * NCH + ch) * NF * CH, *zi = c->Zi + (size_t)(p * NCH + ch) * NF * CH;
        const float *kr = c->Kr + ch * NF * CH, *ki = c->Ki + ch * NF * CH;
        int o0 = 2 * p * T * D + ch * CH, two = 2 * p + 1 < nb;
        mfft(&S->pl, VB + o0, two ? VB + o0 + T * D : NULL, D, zr, zi, s, s + NF * CH);
        for (int k = 0; k < NF * CH; k++) { ur[k] = zr[k] * kr[k] - zi[k] * ki[k]; ui[k] = zr[k] * ki[k] + zi[k] * kr[k]; }
        mifft(&S->pl, ur, ui, y->cc + o0, two ? y->cc + o0 + T * D : NULL, D, 1.f / NF, s, s + NF * CH);
    }
}

// DP[0] = d q (post), cc = d(conv out) -> pre <- d pre, and the mixer's parameter grads
static void mixer_bwd(Stack *S, Layer *y, int nb) {
    int T = S->T, N = S->N, NF = S->pl.NF;
    Conv *c = &y->cv; int n = nb * T, np2 = (nb + 1) / 2;
    #pragma omp parallel for
    for (int ch = 0; ch < NCH; ch++) {             // conv^T: du = g corr h, dh = sum_b g corr u (Re of the
        float *s = SCR + (size_t)TID * 6 * NF * CH, *gr = s + 2 * NF * CH, *gi = gr + NF * CH, *ar = gi + NF * CH, *ai = ar + NF * CH;
        const float *kr = c->Kr + ch * NF * CH, *ki = c->Ki + ch * NF * CH;      // packed product: the cross
        memset(ar, 0, 2 * NF * CH * 4);                                          // terms are imaginary)
        for (int p = 0; p < np2; p++) {
            const float *zr = c->Zr + (size_t)(p * NCH + ch) * NF * CH, *zi = c->Zi + (size_t)(p * NCH + ch) * NF * CH;
            int o0 = 2 * p * T * D + ch * CH, two = 2 * p + 1 < nb;
            mfft(&S->pl, y->cc + o0, two ? y->cc + o0 + T * D : NULL, D, gr, gi, s, s + NF * CH);
            for (int k = 0; k < NF * CH; k++) {
                ar[k] += gr[k] * zr[k] + gi[k] * zi[k]; ai[k] += gi[k] * zr[k] - gr[k] * zi[k];
                float r = gr[k] * kr[k] + gi[k] * ki[k]; gi[k] = gi[k] * kr[k] - gr[k] * ki[k]; gr[k] = r;
            }
            mifft(&S->pl, gr, gi, VB + o0, two ? VB + o0 + T * D : NULL, D, 1.f / NF, s, s + NF * CH);
        }
        mifft(&S->pl, ar, ai, c->dh + ch * CH, NULL, D, 1.f / NF, s, s + NF * CH);
    }
    filter_bwd(S, c);
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {                   // du (+ skip) -> dk, dv
        float *cc = y->cc + t * D, *k = y->post + ((size_t)N + t) * D, *v = k + (size_t)N * D;
        for (int ch = 0; ch < D; ch++) {
            float du = VB[t * D + ch] + cc[ch] * c->be[ch];
            DP[((size_t)N + t) * D + ch] = du * v[ch]; DP[((size_t)2 * N + t) * D + ch] = du * k[ch];
        }
    }
    static float *acc; if (!acc) acc = fa((size_t)NT * 13 * D);   // skip bias and short conv grads: per-thread rows
    #pragma omp parallel
    {
        float *A = acc + (size_t)TID * 13 * D, iu = 1 / c->as[3], su = c->as[3];   // [bias][sb x3][sw j=0..2 x w=0..2]
        memset(A, 0, 13 * D * sizeof(float));
        #pragma omp for
        for (int t = 0; t < n; t++) {
            int tt = t % T; const float *k = y->post + ((size_t)N + t) * D, *v = k + (size_t)N * D, *cc = y->cc + (size_t)t * D;
            for (int ch = 0; ch < D; ch++) {        // the int8 u the forward used (as sq8)
                float r = rintf(k[ch] * v[ch] * iu); r = r > 127.f ? 127.f : r < -127.f ? -127.f : r; A[ch] += cc[ch] * r * su;
            }
            for (int w = 0; w < 3; w++) {
                const float *g = DP + ((size_t)w * N + t) * D; float *sb = A + (1 + w) * D;
                for (int ch = 0; ch < D; ch++) sb[ch] += g[ch];
                for (int j = 0; j < 3 && j <= tt; j++) {
                    const float *p = y->pre + ((size_t)w * N + t - j) * D; float *sw = A + (4 + j * 3 + w) * D;
                    for (int ch = 0; ch < D; ch++) sw[ch] += g[ch] * p[ch];
                }
            }
        }
    }
    #pragma omp parallel for
    for (int ch = 0; ch < D; ch++) {
        float s[13] = {0};
        for (int th = 0; th < NT; th++) for (int i = 0; i < 13; i++) s[i] += acc[((size_t)th * 13 + i) * D + ch];
        c->bias->g[ch] += s[0];
        for (int w = 0; w < 3; w++) { c->sb->g[w * D + ch] += s[1 + w]; for (int j = 0; j < 3; j++) c->sw->g[j * 3 * D + w * D + ch] += s[4 + j * 3 + w]; }
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

static void stack_fwd(Stack *S, int nb, int train) {    // S->X[0] -> S->X[L], nb sequences
    int T = S->T, N = S->N, n = nb * T; float **X = S->X;
    for (int l = 0; l < S->L; l++) {
        Layer *y = &S->ly[l]; Conv *c = &y->cv;
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_prep(ms[i]);
#ifdef VNNI
        for (int i = 0; i < 7; i++) imon_prep(&y->im[i], ms[i]);
#endif
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {           // phase 1 (per token): rms+quant, Mon_q, Mon_k, Mon_v
            if (S->len && t % T >= S->len[t / T]) {   // padding: zeros keep the FFT finite, and causality keeps
                for (int w = 0; w < 3; w++) memset(y->pre + ((size_t)w * N + t) * D, 0, D * sizeof(float));   // it out of valid steps
                continue;
            }
            const float *x = X[l] + t * D; int8_t *q = y->q1 + t * D;
            y->r1[t] = rinv(x); y->s1[t] = q8t(x, q, y->r1[t]);
            for (int w = 0; w < 3; w++) TMON(y, w, ms[w], q, y->s1[t], y->pre + ((size_t)w * N + t) * D, t);
        }
        mixer_fwd(S, y, nb, train);
        #pragma omp parallel for
        for (int t = 0; t < n; t++) {           // phase 2 (per token): gate, Mon_o, +res, rms, GLU, +res
            if (S->len && t % T >= S->len[t / T]) { memset(X[l + 1] + (size_t)t * D, 0, D * sizeof(float)); continue; }
            float g[D], o[D], dd;
            float *cc = y->cc + t * D, *q = y->post + (size_t)t * D, *u = VB + t * D, *x = X[l] + t * D, *x1 = y->x1 + t * D;
            float *ga = y->ga + t * D, *ub = y->ub + t * D;
            for (int ch = 0; ch < D; ch++) g[ch] = q[ch] * (cc[ch] + c->be[ch] * u[ch]);
            y->so[t] = q8t(g, y->qo + t * D, 1);
            TMON(y, 3, &y->o, y->qo + t * D, y->so[t], o, t);
            for (int i = 0; i < D; i++) x1[i] = x[i] + o[i];
            y->r2[t] = rinv(x1); y->s2[t] = q8t(x1, y->q2 + t * D, y->r2[t]);
            TMON(y, 4, &y->g, y->q2 + t * D, y->s2[t], ga, t);
            TMON(y, 5, &y->u, y->q2 + t * D, y->s2[t], ub, t);
            #pragma omp simd private(dd)
            for (int i = 0; i < D; i++) g[i] = gelu(ga[i], &dd) * ub[i];
            y->sd[t] = q8t(g, y->qd + t * D, 1);
            TMON(y, 6, &y->d, y->qd + t * D, y->sd[t], o, t);
            for (int i = 0; i < D; i++) X[l + 1][t * D + i] = x1[i] + o[i];
        }
    }
}

static void stack_bwd(Stack *S, float *DX, int nb) {    // DX: grad wrt X[L] in, grad wrt X[0] out
    int T = S->T, N = S->N, n = nb * T; float **X = S->X;
    for (int l = S->L - 1; l >= 0; l--) {               // DX: grad wrt X[l+1] -> grad wrt X[l]
        Layer *y = &S->ly[l]; Conv *c = &y->cv;
        #pragma omp parallel for
        for (int t0 = 0; t0 < n; t0 += 4) {             // phase A (4 tokens at a time): GLU, rms2, Mon_o, output gate
            int tid = TID; int8_t gc[4][2][4][D];       // grad codes [Mon d, u, g, o][L side, R side][token]
            for (int j = 0; j < 4; j++) {
                int t = t0 + j;
                if (S->len && t % T >= S->len[t / T]) {
                    memset(DP + (size_t)t * D, 0, D * sizeof(float)); memset(y->cc + (size_t)t * D, 0, D * sizeof(float));
                    for (int w = 0; w < 4; w++) memset(gc[w][0][j], 0, D), memset(gc[w][1][j], 0, D);
                    continue;
                }
                float a[D], b[D], e[D], dd, *dx = DX + t * D, *ga = y->ga + t * D, *ub = y->ub + t * D;
                BWDX(y, 6, &y->d, y->sd[t], dx, a, t, tid, gc[0][0][j], gc[0][1][j]);
                #pragma omp simd private(dd)
                for (int i = 0; i < D; i++) { float gl = gelu(ga[i], &dd); b[i] = a[i] * ub[i] * dd; a[i] *= gl; }
                BWDX(y, 5, &y->u, y->s2[t], a, e, t, tid, gc[1][0][j], gc[1][1][j]);
                BWDX(y, 4, &y->g, y->s2[t], b, a, t, tid, gc[2][0][j], gc[2][1][j]);
                for (int i = 0; i < D; i++) a[i] += e[i];
                rms_bwd(y->x1 + t * D, y->r2[t], a, dx);
                BWDX(y, 3, &y->o, y->so[t], dx, a, t, tid, gc[3][0][j], gc[3][1][j]);
                float *cc = y->cc + t * D, *q = y->post + (size_t)t * D, *k = q + (size_t)N * D, *v = k + (size_t)N * D;
                float iu = 1 / c->as[3], su = c->as[3], *dq = DP + (size_t)t * D;
                for (int ch = 0; ch < D; ch++) {        // DP[0] <- dq; cc <- d(conv out)
                    float r = rintf(k[ch] * v[ch] * iu); r = r > 127.f ? 127.f : r < -127.f ? -127.f : r;   // the int8 u
                    dq[ch] = a[ch] * (cc[ch] + c->be[ch] * r * su); cc[ch] = a[ch] * q[ch];                 // the forward used
                }
            }
            mon_dw4(&y->d, tid, t0, y->qd, gc[0][0], gc[0][1]); mon_dw4(&y->u, tid, t0, y->q2, gc[1][0], gc[1][1]);
            mon_dw4(&y->g, tid, t0, y->q2, gc[2][0], gc[2][1]); mon_dw4(&y->o, tid, t0, y->qo, gc[3][0], gc[3][1]);
        }
        mixer_bwd(S, y, nb);
        #pragma omp parallel for
        for (int t0 = 0; t0 < n; t0 += 4) {             // phase B (4 tokens at a time): Mon_q/k/v, rms1
            int tid = TID; int8_t gc[3][2][4][D];
            Mon *ms[] = {&y->q, &y->k, &y->v};
            for (int j = 0; j < 4; j++) {
                int t = t0 + j;
                if (S->len && t % T >= S->len[t / T]) { for (int w = 0; w < 3; w++) memset(gc[w][0][j], 0, D), memset(gc[w][1][j], 0, D); continue; }
                float b[D], sum[D] = {0};
                for (int w = 0; w < 3; w++) {
                    BWDX(y, w, ms[w], y->s1[t], y->pre + ((size_t)w * N + t) * D, b, t, tid, gc[w][0][j], gc[w][1][j]);
                    for (int ch = 0; ch < D; ch++) sum[ch] += b[ch];
                }
                rms_bwd(X[l] + t * D, y->r1[t], sum, DX + t * D);
            }
            for (int w = 0; w < 3; w++) mon_dw4(ms[w], tid, t0, y->q1, gc[w][0], gc[w][1]);
        }
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_reduce(ms[i]);
    }
}

// ---- heads: rms + logits over a [V][D] matrix, int8 x int8 with STE ----------------------------------
// Weights are int8 per output row (absmax), the normed input int8 per token (absmax), so inference runs the
// head as one int8 dot product per logit; training runs the same fake-quantised values in fp32.
static float *HN, *RF, *LOGIT, *S1, *G1, *G2, *WD;   // quantised normed x, its rms, logits (then grad), scratch, grads, W
static void head_quant(const float *W, float *Wd, int8_t *wq, float *sw) {   // per-row int8; Wd: dequantised
    for (int c = 0; c < V; c++) {
        int8_t q[D]; float s = q8t(W + c * D, q, 1);
        for (int i = 0; i < D; i++) { if (Wd) Wd[c * D + i] = q[i] * s; if (wq) wq[c * D + i] = q[i]; }
        if (sw) sw[c] = s;
    }
}
static float *WT;                             // dequantised head transposed [D][V]
#define HB 8                                  // tokens per tile: each weight row is read once per tile, not per token
static void head_fwd(const float *X, int n, const float *W) {
    head_quant(W, WD, NULL, NULL);
    for (int c = 0; c < V; c++) for (int i = 0; i < D; i++) WT[i * V + c] = WD[c * D + i];
    #pragma omp parallel for
    for (int t0 = 0; t0 < n; t0 += HB) {
        float acc[HB][V] = {{0}};
        for (int t = t0; t < t0 + HB; t++) {
            int8_t q[D]; float *h = HN + (size_t)t * D, r = RF[t] = rinv(X + (size_t)t * D), sx = q8t(X + (size_t)t * D, q, r);
            for (int i = 0; i < D; i++) h[i] = q[i] * sx;
        }
        for (int i = 0; i < D; i++) {
            const float *w = WT + i * V;
            for (int j = 0; j < HB; j++) { float hi = HN[(size_t)(t0 + j) * D + i]; for (int c = 0; c < V; c++) acc[j][c] += hi * w[c]; }
        }
        memcpy(LOGIT + (size_t)t0 * V, acc, sizeof acc);
    }
}
static void head_bwd(const float *X, int n, const float *W, float *Wg, float *G) {   // LOGIT holds dLoss/dlogits
    head_quant(W, WD, NULL, NULL);
    #pragma omp parallel for
    for (int t0 = 0; t0 < n; t0 += HB) {                                    // d h = dlogits . W, a tile of tokens per W row
        float s[HB][D] = {{0}};
        for (int c = 0; c < V; c++) {
            const float *w = WD + c * D;
            for (int j = 0; j < HB; j++) { float a = LOGIT[(size_t)(t0 + j) * V + c]; for (int i = 0; i < D; i++) s[j][i] += a * w[i]; }
        }
        for (int j = 0; j < HB; j++) {
            int t = t0 + j; float *g = G + (size_t)t * D;
            for (int i = 0; i < D; i++) g[i] = 0;
            rms_bwd(X + (size_t)t * D, RF[t], s[j], g);
        }
    }
    #pragma omp parallel for
    for (int c0 = 0; c0 < V; c0 += 16) for (int t = 0; t < n; t++) {       // dW += dlogits^T h, 16 rows per h read
        const float *h = HN + (size_t)t * D;
        for (int c = c0; c < c0 + 16; c++) { float a = LOGIT[(size_t)t * V + c], *w = Wg + c * D; for (int i = 0; i < D; i++) w[i] += a * h[i]; }
    }
}
static float xent(const uint8_t *const *tgt, int n, int grad) {   // mean CE (nats/byte); tgt[b][t]
    double loss = 0;
    #pragma omp parallel for reduction(+:loss)
    for (int t = 0; t < n; t++) {
        float *z = LOGIT + (size_t)t * V, mx = z[0], s = 0; int y = tgt[t / TB][t % TB];
        for (int c = 1; c < V; c++) mx = MAXF(mx, z[c]);
        for (int c = 0; c < V; c++) s += expf(z[c] - mx);
        loss += logf(s) + mx - z[y];
        if (grad) { for (int c = 0; c < V; c++) z[c] = expf(z[c] - mx) / s / n; z[y] -= 1.f / n; }
    }
    return loss / n;
}

// ---- entropy model: a small byte-level moth; its next-byte entropies place the patch boundaries ----
static P *Eh;                                 // byte embedding, tied with its head
static Stack SH;
static void hm_fwd(const uint8_t *const *win, int nb, int train) {
    for (int t = 0; t < nb * TB; t++) memcpy(SH.X[0] + (size_t)t * D, Eh->w + win[t / TB][t % TB] * D, D * sizeof(float));
    stack_fwd(&SH, nb, train);
    head_fwd(SH.X[LH], nb * TB, Eh->w);
}
static void hm_bwd(const uint8_t *const *win, int nb) {
    head_bwd(SH.X[LH], nb * TB, Eh->w, Eh->g, G1);
    stack_bwd(&SH, G1, nb);
    for (int t = 0; t < nb * TB; t++) { float *g = Eh->g + win[t / TB][t % TB] * D; for (int i = 0; i < D; i++) g[i] += G1[(size_t)t * D + i]; }
}

// ---- data and patching ------------------------------------------------------------------------------
static uint8_t *data; static long ndata, ntrain;
static float *ENT, theta;                     // ENT[i]: entropy (nats) of byte i given up to TB bytes before it
static long pick(int val) {                   // window offset; >= 8 so every hash n-gram has real bytes
    long lo = val ? ntrain : 8, hi = (val ? ndata : ntrain) - LCTX - 2;
    rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;   // all 64 bits: urand's 24 would quantise offsets past 16 MB
    return lo + (long)(rs % (uint64_t)(hi - lo));
}
static inline float fexp(float x) {           // e^x for x <= 0: 2^floor * a degree-5 polynomial for 2^frac; vectorises
    float t = MAXF(x, -87.f) * 1.44269504f, fl = floor_(t), f = t - fl;
    float p = 1.8775767e-3f; p = p * f + 8.9893397e-3f; p = p * f + 5.5826318e-2f; p = p * f + 2.4015361e-1f; p = p * f + 6.9315308e-1f; p = p * f + 9.9999994e-1f;
    int32_t b = ((int32_t)fl + 127) << 23; float sc; memcpy(&sc, &b, 4);
    return p * sc;
}
static float entropy_of(const float *z) {     // H = log sum e^d - sum e^d d / sum e^d, d = z - max: one exp per byte value
    float mx = z[0], s = 0, sd = 0;
    #pragma omp simd reduction(max:mx)
    for (int c = 0; c < V; c++) mx = z[c] > mx ? z[c] : mx;
    #pragma omp simd reduction(+:s, sd)
    for (int c = 0; c < V; c++) { float d = z[c] - mx, e = fexp(d); s += e; sd += e * d; }
    return logf(s) - sd / s;
}
static void entropies(void) {                 // run the entropy model over the corpus once, windows overlapping by half
    ENT = fa(ndata + 1);
    for (long w0 = 0; w0 + TB <= ndata;) {
        const uint8_t *win[NW]; long st[NW]; int nb = 0;
        for (; nb < NW && w0 + TB <= ndata; nb++, w0 += TB / 2) { st[nb] = w0; win[nb] = data + w0; }
        hm_fwd(win, nb, 0);
        #pragma omp parallel for collapse(2)
        for (int b = 0; b < nb; b++) for (int t = 0; t < TB; t++)
            if (st[b] == 0 || t >= TB / 2 - 1) ENT[st[b] + t + 1] = entropy_of(LOGIT + (size_t)(b * TB + t) * V);
    }
}
static int cmpf(const void *a, const void *b) { float x = *(const float *)a, y = *(const float *)b; return (x > y) - (x < y); }
// s[0..LCTX]: 1 where a patch starts (s[LCTX]: right after the sequence). Returns the sequence's patch count.
static int patchify(long off, uint8_t *s) {
    int np_ = 2, len = 1; s[0] = s[1] = 1;    // BLT: the first patch is a single byte
    for (int t = 2; t <= LCTX; t++) {
        int st = ENT[off + t] > theta || len >= PMAX;
        if (st && t < LCTX && np_ >= TG) st = 0; // at most TG patches in a sequence
        s[t] = st; if (st) { np_ += t < LCTX; len = 1; } else len++;
    }
    return np_;
}

// ---- BLT ------------------------------------------------------------------------------------------------
static P *Eb, *Hs[NG], *Wo;                   // byte embedding, hash n-gram tables, output head
static Stack SE, SG, SD;                      // local encoder, global, local decoder
static int *hidx[NG], *cpb, npat[B];          // hash rows per byte, decoder's patch per byte, patches per sequence
static int16_t *arg;                          // max-pool argmax per patch and channel
static uint8_t sb[B][LCTX + 1];               // patch starts per sequence
static const uint64_t primes[NG] = {1000000007ull, 5915587277ull, 1500450271ull, 3267000013ull, 5754853343ull, 4093082899ull};
static int hrow(const uint8_t *p, int n, int k) {   // BLT's rolling polynomial hash of the n bytes ending at p
    uint64_t h = 0, pw = 1;
    for (int i = 0; i < n; i++) { h += (uint64_t)(p[i - n + 1] + 4) * pw; pw *= primes[k]; }   // + 4: BLT's byte offset
    return (int)(h % HV);
}
static void targets(const long *off, int nb, const uint8_t **tgt) {   // next-byte targets, one row per TB-byte chunk
    for (int b = 0; b < nb; b++) for (int c = 0; c < NCK; c++) tgt[b * NCK + c] = data + off[b] + c * TB + 1;
}
static void blt_fwd(const uint8_t *const *win, const long *off, int nb, int train) {   // win: nb sequences of LCTX bytes
    int n = nb * LCTX;
    for (int b = 0; b < nb; b++) npat[b] = patchify(off[b], sb[b]);
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {             // embeddings: byte + hashed 3..8-grams
        const uint8_t *p = win[t / LCTX] + t % LCTX; float *x = SE.X[0] + (size_t)t * D;
        memcpy(x, Eb->w + *p * D, D * sizeof(float));
        for (int k = 0; k < NG; k++) {
            int r = hidx[k][t] = hrow(p, k + 3, k); const float *e = Hs[k]->w + (size_t)r * D;
            for (int i = 0; i < D; i++) x[i] += e[i];
        }
    }
    stack_fwd(&SE, nb * NCK, train);          // chunks: rows of a sequence stay contiguous
    const float *e = SE.X[LE]; float *g = SG.X[0];
    #pragma omp parallel for
    for (int b = 0; b < nb; b++) {            // patch ids, the decoder's patch, max-pool per patch
        int pid = -1; memset(g + (size_t)b * TG * D, 0, TG * D * sizeof(float));
        for (int t = 0; t < LCTX; t++) {
            pid += sb[b][t];
            cpb[b * LCTX + t] = pid + sb[b][t + 1] - 1;   // the last patch that ends at or before byte t
            float *gk = g + ((size_t)b * TG + pid) * D; int16_t *ak = arg + ((size_t)b * TG + pid) * D; const float *et = e + ((size_t)b * LCTX + t) * D;
            if (sb[b][t]) { memcpy(gk, et, D * sizeof(float)); for (int i = 0; i < D; i++) ak[i] = t; }
            else for (int i = 0; i < D; i++) if (et[i] > gk[i]) { gk[i] = et[i]; ak[i] = t; }
        }
    }
    SG.len = npat; stack_fwd(&SG, nb, train);
    const float *z = SG.X[LG]; float *d = SD.X[0];
    #pragma omp parallel for
    for (int t = 0; t < n; t++) {             // decoder input: encoder state + the global state of its patch
        const float *zt = z + ((size_t)(t / LCTX) * TG + cpb[t]) * D;
        for (int i = 0; i < D; i++) d[(size_t)t * D + i] = e[(size_t)t * D + i] + zt[i];
    }
    stack_fwd(&SD, nb * NCK, train);
    head_fwd(SD.X[LD], n, Wo->w);
}
static void blt_bwd(const uint8_t *const *win, int nb) {
    int n = nb * LCTX;
    head_bwd(SD.X[LD], n, Wo->w, Wo->g, G1);
    stack_bwd(&SD, G1, nb * NCK);                   // G1: grad wrt decoder input = grad wrt e (residual) and z
    #pragma omp parallel for
    for (int b = 0; b < nb; b++) {
        float *gz = G2 + (size_t)b * TG * D; memset(gz, 0, TG * D * sizeof(float));
        for (int t = 0; t < LCTX; t++) { float *gt = gz + (size_t)cpb[b * LCTX + t] * D; const float *g = G1 + ((size_t)b * LCTX + t) * D; for (int i = 0; i < D; i++) gt[i] += g[i]; }
    }
    stack_bwd(&SG, G2, nb);                   // G2: grad wrt pooled patches
    #pragma omp parallel for
    for (int b = 0; b < nb; b++) for (int k = 0; k < npat[b]; k++) for (int i = 0; i < D; i++)
        G1[((size_t)b * LCTX + arg[((size_t)b * TG + k) * D + i]) * D + i] += G2[((size_t)b * TG + k) * D + i];
    stack_bwd(&SE, G1, nb * NCK);
    for (int t = 0; t < n; t++) {
        const float *g = G1 + (size_t)t * D; float *w = Eb->g + win[t / LCTX][t % LCTX] * D;
        for (int i = 0; i < D; i++) w[i] += g[i];
        for (int k = 0; k < NG; k++) { float *h = Hs[k]->g + (size_t)hidx[k][t] * D; for (int i = 0; i < D; i++) h[i] += g[i]; }
    }
}

static void adam(int step, int k0, int k1, int steps) {   // params [k0, k1)
    float lr = step < 100 ? LR * step / 100 : LR * (0.1f + 0.45f * (1 + cosf(3.14159265f * step / steps)));
    float c1 = 1 - powf(0.9f, step), c2 = 1 - powf(0.95f, step);
    for (int k = k0; k < k1; k++) {
        P *p = &ps[k];
        #pragma omp parallel for
        for (int i = 0; i < p->n; i++) {
            float g = p->g[i]; p->g[i] = 0;
            if (step <= 0) continue;
            p->m[i] = 0.9f * p->m[i] + 0.1f * g; p->v[i] = 0.95f * p->v[i] + 0.05f * g * g;
            p->w[i] -= lr * (p->m[i] / c1) / (sqrtf(p->v[i] / c2) + 1e-8f);
        }
    }
}

// ---- inference: one byte at a time, trits x int8 throughout ------------------------------------------
// Each stack keeps per layer the last two q/k/v codes (short conv) and a T-long ring of u codes (long conv);
// the long conv is a direct causal sum over the ternary kernel, skipping all-zero kernel rows. No FFT, no
// window recompute. Per byte: encoder, entropy model and decoder step once; the global stack steps once per
// patch. Embeddings and heads stay fp32.
#if TB <= 256 && TG <= 256
typedef int16_t cacc;                         // |sum of T trit * int8| <= 127 T fits int16 for T <= 256
#else
typedef int32_t cacc;
#endif
static void stack_prep(Stack *S) {            // freeze the trits once
    for (int l = 0; l < S->L; l++) {
        Layer *y = &S->ly[l]; Conv *c = &y->cv;
        Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        for (int i = 0; i < 7; i++) mon_prep(ms[i]);
#ifdef VNNI
        for (int i = 0; i < 7; i++) imon_prep(&y->im[i], ms[i]);
#endif
        filter_fwd(S, c); conv_prep(c);
        y->nh = 0;
        for (int j = 0; j < S->T; j++) { int nz = 0; for (int ch = 0; ch < D; ch++) nz |= c->hq[j * D + ch]; if (nz) y->hrow[y->nh++] = j; }
#ifdef VNNI
        // The long kernel for the 4-tap vpdpbusd, in 4 phase-shifted copies (tap j at group g, slot k, j = 4g + k -
        // phase), cut into 16-channel x 4-tap blocks of 64 trits. Only nonzero blocks are kept (~40%), each packed
        // to 16 bytes, 2 bits a trit (w + 1), in the order one vpmultishiftqb unpacks: output qword q holds trits
        // 8q..8q+7, taken from 2-bit fields (q / 2) * 8 + k of packed qword q % 2.
        int T = S->T, G1 = T / 4 + 1, nb = 0;
        if (!y->kp) { y->uh = malloc((size_t)2 * T * D); y->kp = malloc((size_t)4 * G1 * (D / 16) * 16); y->kg = malloc((size_t)4 * G1 * (D / 16) * sizeof(int)); }
        for (int ph = 0; ph < 4; ph++) for (int c16 = 0; c16 < D / 16; c16++) {
            y->ko[ph][c16] = nb;
            for (int g = 0; g < G1; g++) {
                int8_t w[64]; int nz = 0;
                for (int b = 0; b < 64; b++) { int j = 4 * g + b % 4 - ph, ch = 16 * c16 + b / 4; w[b] = j >= 0 && j < T ? c->hq[j * D + ch] : 0; nz |= w[b]; }
                if (!nz) continue;
                uint8_t *pk = y->kp + (size_t)nb * 16; memset(pk, 0, 16);
                for (int b = 0; b < 64; b++) { int q = b / 8, k = b % 8, bit = ((q / 2) * 8 + k) * 2; pk[(q % 2) * 8 + bit / 8] |= (uint8_t)((w[b] + 1) << (bit % 8)); }
                y->kg[nb++] = g;
            }
        }
        for (int ph = 0; ph < 4; ph++) y->ko[ph][D / 16] = ph < 3 ? y->ko[ph + 1][0] : nb;   // (end of each phase's list)
        for (int ch = 0; ch < D; ch++) { int sum = 0; for (int j = 0; j < T; j++) sum += c->hq[j * D + ch]; y->hc[ch] = 128 * sum; }
#endif
    }
}
static void stack_reset(Stack *S) {
    for (int l = 0; l < S->L; l++) {
        memset(S->ly[l].hist, 0, sizeof S->ly[l].hist); memset(S->ly[l].ring, 0, (size_t)S->T * D);
#ifdef VNNI
        memset(S->ly[l].uh, 0x80, (size_t)2 * S->T * D);   // u = 0
#endif
    }
    S->ipos = 0;
}
static void glu(float *restrict g, const float *restrict f) {   // g *= gelu(f)
    for (int i = 0; i < D; i++) g[i] *= gelu0(f[i]);
}
static void stack_step(Stack *S, float *x) {  // x [D]: input in, output out
    float x1[D], o[D], f[D], g[D]; int8_t q[D], cur[3][D]; long p = S->ipos++; int T = S->T;
    for (int l = 0; l < S->L; l++) {
        Layer *y = &S->ly[l]; Conv *c = &y->cv;
        Mon *ms[] = {&y->q, &y->k, &y->v};
        float s1 = q8t(x, q, rinv(x));
        for (int w = 0; w < 3; w++) { IMON(y, w, ms[w], q, s1, f); float is = 1 / c->as[w]; for (int ch = 0; ch < D; ch++) cur[w][ch] = sc8(f[ch], is); }
        float post[3][D];
        for (int w = 0; w < 3; w++) {         // ternary short conv over the last three codes
            const int8_t *w0 = c->swq + w * D, *w1 = w0 + 3 * D, *w2 = w1 + 3 * D, *h0 = y->hist[0][w], *h1 = y->hist[1][w];
            float sc = c->sws[w] * c->as[w];
            for (int ch = 0; ch < D; ch++) post[w][ch] = (w0[ch] * cur[w][ch] + w1[ch] * h0[ch] + w2[ch] * h1[ch]) * sc + c->sbe[w * D + ch];
        }
        memcpy(y->hist[1], y->hist[0], sizeof y->hist[0]); memcpy(y->hist[0], cur, sizeof cur);
        int8_t *u = y->ring + (p % T) * D;
        float iu = 1 / c->as[3]; for (int ch = 0; ch < D; ch++) u[ch] = sc8(post[1][ch] * post[2][ch], iu);
#ifdef VNNI
        // long conv on vpdpbusd, 4 taps per lane. History runs backwards: u[p - j] sits at position r + j, r =
        // T - 1 - p mod T, stored twice (r and r + T) so every window is contiguous. Positions are grouped by 4
        // per channel; the kernel copy matching r's alignment lines taps up with them.
        int32_t acc[D];
        {
            int r0 = T - 1 - (int)(p % T), ph = r0 & 3, G0 = r0 >> 2;
            static const uint8_t sp[64] = {0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7,
                                           8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11, 12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15};
            const __m512i spread = _mm512_loadu_si512(sp), flip = _mm512_set1_epi8((char)0x80);
            for (int copy = 0; copy < 2; copy++) {   // write u_p + 128 into slot (pos & 3) of each channel at pos = r0, r0 + T
                int pos = r0 + copy * T; uint8_t *dst = y->uh + (size_t)(pos >> 2) * D * 4;
                __mmask64 mk = 0x1111111111111111ull << (pos & 3);
                for (int c0 = 0; c0 < D; c0 += 16) {
                    __m512i v = _mm512_xor_si512(_mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(u + c0))), flip);
                    _mm512_mask_storeu_epi8(dst + c0 * 4, mk, _mm512_permutexvar_epi8(spread, v));
                }
            }
            static const uint8_t ms[64] = {0,2,4,6,8,10,12,14, 0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30, 16,18,20,22,24,26,28,30,
                                           32,34,36,38,40,42,44,46, 32,34,36,38,40,42,44,46, 48,50,52,54,56,58,60,62, 48,50,52,54,56,58,60,62};
            const __m512i sel = _mm512_loadu_si512(ms), three = _mm512_set1_epi8(3), one = _mm512_set1_epi8(1);
            #define KUNP(i) _mm512_sub_epi8(_mm512_and_si512(_mm512_multishift_epi64_epi8(sel, _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(y->kp + (size_t)(i) * 16)))), three), one)
            #define HIST(i) _mm512_loadu_si512(y->uh + (size_t)(G0 + y->kg[i]) * D * 4 + c * 64)
            for (int c = 0; c < D / 16; c++) {    // per chunk, only its nonzero blocks; 4 accumulators hide the latency
                __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0; int i = y->ko[ph][c], e = y->ko[ph][c + 1];
                for (; i + 4 <= e; i += 4) {
                    a0 = _mm512_dpbusd_epi32(a0, HIST(i), KUNP(i)); a1 = _mm512_dpbusd_epi32(a1, HIST(i + 1), KUNP(i + 1));
                    a2 = _mm512_dpbusd_epi32(a2, HIST(i + 2), KUNP(i + 2)); a3 = _mm512_dpbusd_epi32(a3, HIST(i + 3), KUNP(i + 3));
                }
                for (; i < e; i++) a0 = _mm512_dpbusd_epi32(a0, HIST(i), KUNP(i));
                __m512i sum = _mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3));
                _mm512_storeu_si512(acc + 16 * c, _mm512_sub_epi32(sum, _mm512_loadu_si512(y->hc + 16 * c)));
            }
            #undef KUNP
            #undef HIST
        }
#else
        cacc acc[D] = {0};                    // long conv: sum_j h[j] u[p - j] over the nonzero kernel rows
        for (int r = 0; r < y->nh && y->hrow[r] <= p; r++) {
            int j = y->hrow[r]; const int8_t *hj = c->hq + j * D, *uj = y->ring + ((p - j) % T) * D;
            for (int ch = 0; ch < D; ch++) acc[ch] += hj[ch] * uj[ch];
        }
#endif
        for (int ch = 0; ch < D; ch++) g[ch] = post[0][ch] * (acc[ch] * c->hs[ch] + c->be[ch] * u[ch]) * c->as[3];
        float so = q8t(g, q, 1); IMON(y, 3, &y->o, q, so, o);
        for (int i = 0; i < D; i++) x1[i] = x[i] + o[i];
        float s2 = q8t(x1, q, rinv(x1)); IMON(y, 4, &y->g, q, s2, f); IMON(y, 5, &y->u, q, s2, g);
        glu(g, f);
        float sd = q8t(g, q, 1); IMON(y, 6, &y->d, q, sd, o);
        for (int i = 0; i < D; i++) x[i] = x1[i] + o[i];
    }
}
typedef struct { int8_t q[V * D]; uint8_t p[V * D]; float s[V]; } IHead;   // head: int8 rows, VNNI layout, scales
static IHead ih_ent, ih_out;
static void ihead_prep(IHead *h, const float *W) {
    head_quant(W, NULL, h->q, h->s);
    for (int k = 0; k < D / 4; k++) for (int c = 0; c < V; c++) for (int j = 0; j < 4; j++)   // [i/4][c][i%4], + 128: u8
        h->p[(k * V + c) * 4 + j] = (uint8_t)(h->q[c * D + 4 * k + j] + 128);
}
static void head_step(const float *x, const IHead *h, float *logits) {   // int8 dot per logit, exact
    int8_t q[D]; float sx = q8t(x, q, rinv(x));
#ifdef VNNI
    __m512i acc[V / 16], sum = _mm512_setzero_si512();                  // (w + 128) . x - 128 sum x
    for (int c = 0; c < V / 16; c++) acc[c] = _mm512_setzero_si512();
    for (int k = 0; k < D / 4; k++) {
        int32_t x4; memcpy(&x4, q + 4 * k, 4); __m512i xv = _mm512_set1_epi32(x4);
        for (int c = 0; c < V / 16; c++) acc[c] = _mm512_dpbusd_epi32(acc[c], _mm512_loadu_si512(h->p + (k * V + 16 * c) * 4), xv);
        sum = _mm512_dpbusd_epi32(sum, _mm512_set1_epi8(1), xv);
    }
    __m512i s128 = _mm512_slli_epi32(sum, 7); __m512 vs = _mm512_set1_ps(sx);
    for (int c = 0; c < V / 16; c++)
        _mm512_storeu_ps(logits + 16 * c, _mm512_mul_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_sub_epi32(acc[c], s128)), vs), _mm512_loadu_ps(h->s + 16 * c)));
#else
    for (int c = 0; c < V; c++) { int32_t a = 0; const int8_t *w = h->q + c * D; for (int i = 0; i < D; i++) a += w[i] * q[i]; logits[c] = a * sx * h->s[c]; }
#endif
}
static struct { uint8_t hist[8]; float pm[D], z[D]; int plen, start; long pos, npatch; } bs;
static void blt_reset(const uint8_t *prev8) { // prev8: the 8 bytes before the stream (hash context), or NULL
    stack_reset(&SH); stack_reset(&SE); stack_reset(&SG); stack_reset(&SD);
    memset(&bs, 0, sizeof bs); bs.start = 1;
    if (prev8) memcpy(bs.hist, prev8, 8);
}
// consume byte, return logits for the next one. force: -1 decides the patch boundary from the entropy model,
// 0 / 1 imposes it (to replay the training forward's patches exactly)
static void blt_step(int byte, int force, float *logits) {
    float e[D], h[D], d[D];
    if (NCK > 1 && bs.pos && bs.pos % TB == 0) { stack_reset(&SE); stack_reset(&SD); }   // a new chunk
    memmove(bs.hist, bs.hist + 1, 7); bs.hist[7] = byte;
    memcpy(e, Eb->w + byte * D, sizeof e);
    for (int k = 0; k < NG; k++) { const float *r = Hs[k]->w + (size_t)hrow(bs.hist + 7, k + 3, k) * D; for (int i = 0; i < D; i++) e[i] += r[i]; }
    stack_step(&SE, e);
    if (bs.start) { memcpy(bs.pm, e, sizeof e); bs.plen = 1; }        // max-pool the running patch
    else { for (int i = 0; i < D; i++) bs.pm[i] = MAXF(bs.pm[i], e[i]); bs.plen++; }
    memcpy(h, Eh->w + byte * D, sizeof h); stack_step(&SH, h);        // entropy of the next byte
    head_step(h, &ih_ent, logits);
    int st = force >= 0 ? force : bs.pos == 0 || entropy_of(logits) > theta || bs.plen >= PMAX;
    if (st) { memcpy(h, bs.pm, sizeof h); stack_step(&SG, h); memcpy(bs.z, h, sizeof h); bs.npatch++; }   // patch done
    bs.start = st;
    for (int i = 0; i < D; i++) d[i] = e[i] + bs.z[i];
    stack_step(&SD, d);
    head_step(d, &ih_out, logits);
    bs.pos++;
}

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void loss_by_position(int nbatch) {   // validation CE by byte position in the window: does context help?
    static const int edge[] = {0, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384};
    double sum[11] = {0}; long cnt[11] = {0}; const uint8_t *win[B]; long off[B];
    for (int k = 0; k < nbatch; k++) {
        for (int b = 0; b < B; b++) { off[b] = pick(1); win[b] = data + off[b]; }
        blt_fwd(win, off, B, 0);
        for (int t = 0; t < B * LCTX; t++) {
            const float *z = LOGIT + (size_t)t * V; float mx = z[0], s = 0; int pos = t % LCTX, i = 0;
            for (int c = 1; c < V; c++) mx = MAXF(mx, z[c]);
            for (int c = 0; c < V; c++) s += expf(z[c] - mx);
            while (pos >= edge[i + 1]) i++;
            sum[i] += logf(s) + mx - z[data[off[t / LCTX] + pos + 1]]; cnt[i]++;
        }
    }
    printf("val loss by position (nats/byte):");
    for (int i = 0; i < 11 && edge[i] < LCTX; i++) printf(" %d-%d %.3f |", edge[i], edge[i + 1] - 1, sum[i] / cnt[i]);
    printf("\n");
}
static float blt_val(void) {
    float vl = 0; const uint8_t *win[B], *tgt[NW]; long off[B];
    for (int k = 0; k < 8; k++) {
        for (int b = 0; b < B; b++) { off[b] = pick(1); win[b] = data + off[b]; }
        targets(off, B, tgt); blt_fwd(win, off, B, 0); vl += xent(tgt, B * LCTX, 0) / 8;
    }
    return vl;
}

// ---- checkpoints: the whole training state, so a run can resume bit-exactly or just generate ------------
// Layout: header (magic, version, the architecture's compile-time sizes), then the step, RNG and entropy
// threshold, then every parameter in build order (weights, Adam m, Adam v), then the per-layer static int8
// scales (EMA amax, which inference also uses) and the Monarchs' delayed gradient scales.
#define CKMAGIC 0x48544F4Du                   // "MOTH"
#ifndef CKEVERY
#define CKEVERY 5000                          // training steps between checkpoints
#endif
static Stack *const STK[] = {&SH, &SE, &SG, &SD};
static int ckio(FILE *f, int wr, int *step) {  // returns 0 if every field transferred and matched
    #define IO(p, n) do { if ((wr ? fwrite(p, sizeof *(p), n, f) : fread(p, sizeof *(p), n, f)) != (size_t)(n)) return 1; } while (0)
    int32_t h[] = {CKMAGIC, LCTX == TB ? 1 : 2, M, TB, LE, LG, LD, LH, HV, NG, V, PMAX, np}, g[13];   // version 2: + LCTX
    if (wr) IO(h, 13); else { IO(g, 13); for (int i = 0; i < 13; i++) if (g[i] != h[i]) {
        const char *nm[] = {"magic", "version (2 = a -DLCTX build)", "M", "TB", "LE", "LG", "LD", "LH", "HV", "NG", "V", "PMAX", "param count"};
        fprintf(stderr, "checkpoint %s is %d, this build has %d: rebuild with the same -D flags\n", nm[i], g[i], h[i]); return 1; } }
    if (h[1] == 2) { int32_t lc = LCTX, lg = LCTX; if (wr) IO(&lc, 1); else { IO(&lg, 1); if (lg != lc) {
        fprintf(stderr, "checkpoint LCTX is %d, this build has %d: rebuild with the same -D flags\n", lg, lc); return 1; } } }
    int32_t st = *step; float ps_ = PSZ;
    IO(&st, 1); IO(&rs, 1); IO(&theta, 1); IO(&ps_, 1); *step = st;
    for (int k = 0; k < np; k++) {
        int32_t n = ps[k].n; IO(&n, 1);
        if (n != ps[k].n) { fprintf(stderr, "checkpoint param %d has %d values, expected %d\n", k, n, ps[k].n); return 1; }
        IO(ps[k].w, n); IO(ps[k].m, n); IO(ps[k].v, n);
    }
    for (int s = 0; s < 4; s++) for (int l = 0; l < STK[s]->L; l++) {
        Layer *y = &STK[s]->ly[l]; Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
        IO(y->cv.amax, 4);
        for (int i = 0; i < 7; i++) { IO(&ms[i]->gl.prev, 1); IO(&ms[i]->gr.prev, 1); }
    }
    return 0;
    #undef IO
}
static void ck_save(const char *path, int step) {  // write a temp file, then rename: a crash never leaves half a checkpoint
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
    if (!f || ckio(f, 1, &step) | fclose(f) || rename(tmp, path)) { fprintf(stderr, "could not write checkpoint %s\n", path); exit(1); }
}
static int ck_load(const char *path) {         // returns the step the checkpoint was saved after
    int step = 0; FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "no checkpoint %s\n", path); exit(1); }
    if (ckio(f, 0, &step)) { fprintf(stderr, "bad or incompatible checkpoint %s\n", path); exit(1); }
    fclose(f); return step;
}

int main(int argc, char **argv) {
    // moth [data | -] [-o ckpt] [-r ckpt]       train; "-" reads the corpus from a pipe. -o: save a checkpoint
    //                                             every CKEVERY steps and at the end. -r: resume from one
    // moth -g ckpt [-p prompt] [-n bytes]        no training: load a checkpoint and generate, e.g. after a prompt
    // moth -g ckpt -q prompts.txt               answer each line (\n escaped as a backslash-n) greedily, up to a
    //                                            newline: one answer per line, for scoring against known answers
    const char *path = "input.txt", *out = NULL, *res = NULL, *gen = NULL, *prompt = "\n", *qs = NULL; int ngen = 4000;
    for (int a = 1; a < argc; a++) {
        if (argv[a][0] == '-' && argv[a][1] && !argv[a][2] && strchr("orgpnq", argv[a][1])) {
            if (a + 1 == argc) { fprintf(stderr, "%s needs a value\n", argv[a]); return 1; }
            const char *v = argv[++a];
            switch (argv[a - 1][1]) { case 'o': out = v; break; case 'r': res = v; break; case 'g': gen = v; break;
                                      case 'p': prompt = v; break; case 'n': ngen = atoi(v); break; case 'q': qs = v; break; }
        } else path = argv[a];
    }
    if (!*prompt || ngen < 1) { fprintf(stderr, "empty prompt or no bytes to generate\n"); return 1; }
    const char *ck = gen ? gen : res;
    if (!gen) {
        int in = !strcmp(path, "-");            // "-": read the corpus from a pipe, e.g. a download
        FILE *f = in ? stdin : fopen(path, "rb");
        if (!f) f = fopen(__FILE__, "rb");      // no data? learn to write moth.c
        if (!f) { fprintf(stderr, "no input\n"); return 1; }
        for (long cap = 0, r; ; ndata += r) {
            if (ndata == cap) data = realloc(data, cap = cap ? 2 * cap : 1 << 24);
            if ((r = fread(data + ndata, 1, cap - ndata, f)) == 0) break;
        }
        if (!in) fclose(f);
        if (ndata < 16 * LCTX) { fprintf(stderr, "input too small\n"); return 1; }   // val (last 10%) must hold a sequence
        ntrain = ndata * 9 / 10;
    }
    NT = omp_get_max_threads(); if (NT > 64) NT = 64;
    int NMAX = B * LCTX;
    SCR = fa((size_t)NT * 6 * 2 * (TB > TG ? TB : TG) * CH); VB = fa((size_t)NMAX * D); DP = fa((size_t)3 * NMAX * D);
    HN = fa((size_t)NMAX * D); RF = fa(NMAX); LOGIT = fa((size_t)NMAX * V); S1 = fa((size_t)NMAX * D);
    G1 = fa((size_t)NMAX * D); WD = fa((size_t)V * D); WT = fa((size_t)V * D); G2 = fa((size_t)B * TG * D); arg = calloc((size_t)B * TG * D, 2); cpb = calloc(NMAX, sizeof(int));
    for (int k = 0; k < NG; k++) hidx[k] = calloc(NMAX, sizeof(int));
    const uint8_t *win[NW], *tgt[NW]; long off[NW]; double t0;

    // 1. entropy model -------------------------------------------------------------------------------
    Eh = param(V * D, 0.02f); stack_build(&SH, TB, LH, 1000, NW); int h1 = np;
    if (!ck) {
    printf("entropy model: %d layer(s), d %d, %d bytes of context, %d threads\n", LH, D, TB, NT);
    t0 = now();
    for (int step = -1; step <= HSTEPS; step++) {
        seed = hash(step + 2);
        for (int b = 0; b < NW; b++) { off[b] = pick(0); win[b] = data + off[b]; tgt[b] = win[b] + 1; }
        hm_fwd(win, NW, 1); float loss = xent(tgt, NW * TB, 1); hm_bwd(win, NW); adam(step, 0, h1, HSTEPS);
        if (step % 250 == 0 && step > 0) { printf("  step %4d | loss %.4f | %.0f ms/step\n", step, loss, (now() - t0) * 1e3 / 250); fflush(stdout); t0 = now(); }
    }
    { float vl = 0; for (int k = 0; k < 8; k++) { for (int b = 0; b < NW; b++) { off[b] = pick(1); win[b] = data + off[b]; tgt[b] = win[b] + 1; } hm_fwd(win, NW, 0); vl += xent(tgt, NW * TB, 0) / 8; }
      printf("  val loss %.4f nats/byte\n", vl); }
    }

    // 2. BLT --------------------------------------------------------------------------------------------
    int b0 = np;
    Eb = param(V * D, 0.02f); for (int k = 0; k < NG; k++) Hs[k] = param(HV * D, 0.02f); Wo = param(V * D, 0.02f);
    int bt = np;
    stack_build(&SE, TB, LE, 2000, NW); stack_build(&SG, TG, LG, 3000, B); stack_build(&SD, TB, LD, 4000, NW);
    long nemb = 0, nall = 0; for (int k = b0; k < bt; k++) nemb += ps[k].n; for (int k = b0; k < np; k++) nall += ps[k].n;
    long trits = (long)(LE + LG + LD) * (7 * 2 * W3 + 13 * D) + (long)(LE + LD) * TB * D + (long)LG * TG * D;
    if (NCK > 1) printf("long global context: %d-byte sequences, byte stacks in %d-byte chunks, %d patch slots\n", LCTX, TB, TG);
    printf("BLT: encoder %d, global %d, decoder %d layers, d %d; %.2fM params to train (%.2fM embeddings, of which %.2fM hash n-grams); inference on %.2fM trits\n",
           LE, LG, LD, D, nall / 1e6, nemb / 1e6, NG * HV * D / 1e6, trits / 1e6);
    int step0 = -1;
    if (ck) { step0 = ck_load(ck) + 1; printf("loaded %s: step %d of %d, threshold %.3f nats\n", ck, step0 - 1, STEPS, theta); }
    if (!gen) {
    t0 = now(); entropies();
    if (!res) { long n = ndata - 1; float *tmp = malloc(n * sizeof(float)); memcpy(tmp, ENT + 1, n * sizeof(float)); qsort(tmp, n, sizeof(float), cmpf);
      theta = tmp[(long)((1 - 1 / PSZ) * n)]; free(tmp); }
    { static uint8_t s[LCTX + 1]; double bytes = 0, pat = 0; uint64_t r0 = rs; for (int k = 0; k < 2000; k++) { pat += patchify(pick(0), s); bytes += LCTX; }
      if (res) rs = r0;                        // a resumed run draws the same windows as an unbroken one
      printf("entropies over %ld bytes in %.1f s; threshold %.3f nats -> mean patch %.2f bytes\n", ndata, now() - t0, theta, bytes / pat); }
    t0 = now();
    for (int step = step0; step <= STEPS; step++) {
        seed = hash(step + 7);
        for (int b = 0; b < B; b++) { off[b] = pick(0); win[b] = data + off[b]; }
        targets(off, B, tgt); blt_fwd(win, off, B, 1); float loss = xent(tgt, B * LCTX, 1); blt_bwd(win, B); adam(step, b0, np, STEPS);
        if (step % 50 == 0 && step > 0) { printf("step %4d | loss %.4f | %.0f ms/step\n", step, loss, (now() - t0) * 1e3 / 50); fflush(stdout); t0 = now(); }
        if (step % 500 == 0 && step > 0) { float vl = blt_val(); printf("step %4d | val loss %.4f nats/byte (%.3f bits/byte)\n", step, vl, vl / logf(2)); t0 = now(); }
        if (out && step > 0 && (step % CKEVERY == 0 || step == STEPS)) { ck_save(out, step); printf("saved %s at step %d\n", out, step); fflush(stdout); t0 = now(); }
    }
    }

    if (!gen) loss_by_position(32);

    // 3. inference ------------------------------------------------------------------------------------------
    stack_prep(&SH); stack_prep(&SE); stack_prep(&SG); stack_prep(&SD); ihead_prep(&ih_ent, Eh->w); ihead_prep(&ih_out, Wo->w);
    float lg[V], err = 0, mag = 0; int agree = 0;
    if (!gen) {
    off[0] = pick(1); win[0] = data + off[0];                          // replay a sequence with the training patches
    blt_fwd(win, off, 1, 0);
    blt_reset(win[0] - 8);
    for (int t = 0; t < LCTX; t++) {
        blt_step(win[0][t], sb[0][t + 1], lg);
        for (int c = 0; c < V; c++) { err = fmaxf(err, fabsf(lg[c] - LOGIT[(size_t)t * V + c])); mag = fmaxf(mag, fabsf(LOGIT[(size_t)t * V + c])); }
    }
    blt_reset(win[0] - 8);                                             // same window, patching online
    for (int t = 0; t < LCTX; t++) { int np0 = bs.npatch; blt_step(win[0][t], -1, lg); agree += (bs.npatch > np0) == sb[0][t + 1]; }
    printf("engine vs training forward: max |dlogit| %.1e (logits up to %.1f); online patch boundaries agree on %d/%d bytes\n", err, mag, agree, LCTX);
    }
    if (gen && qs) {                                                   // batch of questions, greedy answers
        FILE *qf = fopen(qs, "rb"); if (!qf) { fprintf(stderr, "no prompts %s\n", qs); return 1; }
        static char ln[1 << 16]; int nq = 0; t0 = now(); long nb = 0;
        while (fgets(ln, sizeof ln, qf)) {
            int n = 0; for (int i = 0; ln[i] && ln[i] != '\n'; i++) ln[n++] = ln[i] == '\\' && ln[i + 1] == 'n' ? (i++, '\n') : ln[i];
            if (!n) continue;
            blt_reset(NULL); blt_step('\n', -1, lg);                    // documents in training follow a blank line
            for (int k = 0; k < n; k++) blt_step((uint8_t)ln[k], -1, lg);
            for (int k = 0; k < 120; k++) {
                int b = 0; for (int j = 1; j < V; j++) if (lg[j] > lg[b]) b = j;
                if (b == '\n') break;
                putchar(b); blt_step(b, -1, lg); nb++;
            }
            putchar('\n'); nq++; nb += n;
        }
        fclose(qf); fprintf(stderr, "answered %d prompts in %.2f s (%.0f bytes/s)\n", nq, now() - t0, nb / (now() - t0));
        return 0;
    }
    char *o = malloc(ngen + 1); int np_ = (int)strlen(prompt), c = (uint8_t)prompt[np_ - 1];
    blt_reset(NULL);
    for (int k = 0; k < np_ - 1; k++) blt_step((uint8_t)prompt[k], -1, lg);   // read the prompt; sampling starts after it
    long p0 = bs.npatch; t0 = now();
    for (int k = 0; k < ngen; k++) {
        blt_step(c, -1, lg);
        float mx = -1e30f, s = 0, r;
        for (int j = 0; j < V; j++) mx = fmaxf(mx, lg[j]);
        for (int j = 0; j < V; j++) s += (lg[j] = expf((lg[j] - mx) / 0.8f));
        c = 0; r = urand() * s; while (c < V - 1 && (r -= lg[c]) > 0) c++;
        o[k] = c;
    }
    t0 = now() - t0; o[ngen] = 0;
    printf("generated %d bytes in %.3f s: %.0f bytes/s, one thread; %ld patches (%.2f bytes each)\n", ngen, t0, ngen / t0, bs.npatch - p0, (double)ngen / (bs.npatch - p0));
    if (gen) { fputs(prompt, stdout); fwrite(o, 1, ngen, stdout); putchar('\n'); } else printf("%.600s\n", o);
    return 0;
}
