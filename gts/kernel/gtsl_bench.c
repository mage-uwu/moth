// Golden Tree Snake (GTS) fork, 2026.
// CPU inference for GTS-L, the attention-free ternary masked LM distilled from ModernBERT-large
// (mamba_ssm/models/gts_l.py). Loads the file scripts/export_gtsl.py writes, checks its logits at the test's masked
// positions against PyTorch's, then times encoding whole sequences and breaks the time down by part. "synth" builds a
// random model of the given shape instead (no file, no check), to time full-size shapes.
//
//   gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp kernel/gtsl_bench.c -o kernel/gtsl_bench -lm
//   OMP_NUM_THREADS=4 ./kernel/gtsl_bench gtsl.bin [repeats]
//   OMP_NUM_THREADS=4 ./kernel/gtsl_bench synth [layers] [linear_rank]      (GTS-L: 28 layers, width 1024)
//
// Each layer: h += BiSSD(LN1 h); h += trees(LN2 h) [+ up(down(LN2 h)) + bias]. LayerNorms subtract the mean (as
// ModernBERT's). Every ternary projection's input is quantised to int8 per token (as trained) and its weights are
// packed two bits per weight (ar_bench.c's TMat); dot products are integer (VNNI when present).
//   BiSSD: in_proj gives each head's C, B (state 64) and x (64); dt = softplus(dt_proj u) in float; per head and
//   direction a 64 x 64 state runs along the sequence: forward decays, writes dt B x^T, then reads C . S (the token's
//   own term included); backward decays, reads, then writes (excluded), so the two sum to the model's mixing matrix.
//   y = forward + backward + D x, then out_proj. Tokens are split across threads for the projections, (head,
//   direction) pairs for the scans.
//   Deep trees: each token walks one root-to-leaf path per tree; only visited rows are read (enc_bench.c's walk).
//   Linear path (rank R > 0): down (R x D) then up (D x R), both ternary with int8 inputs, plus a float bias.
#define main ar_bench_main
#include "ar_bench.c"
#undef main
#ifdef _OPENMP
#include <omp.h>
#endif

#if !HAVE_I8
#error "gtsl_bench needs AVX-512 (F and BW)"
#endif

// Output address of token t, row r: token-major (ldo > 0), or with ldo == 0 "panels": every 64 rows are a contiguous
// T x 64 block, so a head's C, B or x (and the scans' outputs) for consecutive tokens are consecutive in memory.
#define OUTPTR(out, ldo, T, t, r) ((ldo) > 0 ? (out) + (size_t)(t) * (ldo) + (r) : (out) + ((size_t)((r) / 64) * (T) + (t)) * 64 + (r) % 64)
// ------------------------------------------------------------------------------- ternary GEMM (VNNI)
// The dense projections run over all tokens at once. Weights as int8 codes {-1, 0, +1}, laid out in blocks of 16
// rows x 4 inputs (one 64-byte register: lane i holds row i's 4 codes), so one vpdpbusd multiplies 4 inputs of one
// token into 16 rows. Activations are the int8 codes + 128 as unsigned bytes (vpdpbusd is unsigned x signed); the
// 128 x (sum of a row group's codes) this adds is subtracted at the end. A tile is 32 rows x 8 tokens: each weight
// register serves 8 tokens. Integer sums run over one scale group (128 inputs), then are scaled into float.
typedef struct { int rows, rows_pad, K, gsz, ng; signed char *w; float *scale, *off; } TG;

static void tg_from_tm(TG *g, const TMat *m) {  // from a packed TMat: same codes and scales
    g->rows = m->rows; g->rows_pad = (m->rows + 31) / 32 * 32; g->K = m->cols; g->gsz = m->gch * 16; g->ng = m->ng;
    // Every row's groups share one scale (weights trained with one scale per row): one group of K inputs. A group whose
    // codes are all zero packs with scale 0; it does not count against this.
    int one = 1;
    float *rs = (float *)xalloc((size_t)m->rows * 4);
    for (int r = 0; r < m->rows && one; r++)
        for (int gi = 0; gi < m->ng; gi++) {
            const float sc = m->scale[(size_t)r * m->ng + gi];
            if (sc == 0) continue;
            if (rs[r] == 0) rs[r] = sc; else if (sc != rs[r]) { one = 0; break; }
        }
    const int ng_src = m->ng;
    if (one) { g->gsz = g->K; g->ng = 1; }
    const int K4 = g->K / 4;
    g->w = (signed char *)xalloc((size_t)g->rows_pad * g->K);
    g->scale = (float *)xalloc((size_t)g->rows_pad * g->ng * 4); g->off = (float *)xalloc((size_t)g->rows_pad * 4);
    for (int r = 0; r < m->rows; r++) {
        float off = 0;
        for (int gi = 0; gi < g->ng; gi++) {
            int csum = 0;
            (void)ng_src;
            for (int k = gi * g->gsz; k < (gi + 1) * g->gsz; k++) {
                const int bit = k % 16, ch = k / 16;
                const int c = (m->pos[(size_t)r * m->nch + ch] >> bit & 1) - (m->neg[(size_t)r * m->nch + ch] >> bit & 1);
                g->w[(((size_t)(r / 16) * K4 + k / 4) * 16 + r % 16) * 4 + k % 4] = (signed char)c;
                csum += c;
            }
            const float sc = one ? rs[r] : m->scale[(size_t)r * ng_src + gi];
            g->scale[((size_t)(r / 16) * g->ng + gi) * 16 + r % 16] = sc;
            off += 128.0f * csum * sc;
        }
        g->off[r] = off;
    }
    free(rs);
}

// out[t * ldo + r] (=, or += with acc) step[t] * sum_k w[r, k] x[t, k], for x the int8 codes (xs, ldx bytes per token)
static void tg_gemm_vnni(const TG *g, const signed char *xs, int ldx, const float *step, int T, float *out, int ldo, int acc) {
    const unsigned char *xu = (const unsigned char *)xs;
    const int K4 = g->K / 4, gs4 = g->gsz / 4, nrb = g->rows_pad / 32, ntt = (T + 7) / 8;
#pragma omp parallel for collapse(2) schedule(static)
    for (int rb = 0; rb < nrb; rb++)
        for (int tt = 0; tt < ntt; tt++) {
            const int t0 = tt * 8;
            const unsigned char *xp[8];
            for (int j = 0; j < 8; j++) xp[j] = xu + (size_t)(t0 + j < T ? t0 + j : T - 1) * ldx;
            const signed char *w0p = g->w + (size_t)(2 * rb) * K4 * 64, *w1p = g->w + (size_t)(2 * rb + 1) * K4 * 64;
            __m512 f0[8], f1[8];
            for (int j = 0; j < 8; j++) { f0[j] = _mm512_setzero_ps(); f1[j] = _mm512_setzero_ps(); }
            for (int gi = 0; gi < g->ng; gi++) {
                __m512i a0[8], a1[8];
                for (int j = 0; j < 8; j++) { a0[j] = _mm512_setzero_si512(); a1[j] = _mm512_setzero_si512(); }
                for (int ks = gi * gs4; ks < (gi + 1) * gs4; ks++) {
                    const __m512i w0 = _mm512_load_si512(w0p + (size_t)ks * 64), w1 = _mm512_load_si512(w1p + (size_t)ks * 64);
                    for (int j = 0; j < 8; j++) {
                        int v; memcpy(&v, xp[j] + 4 * ks, 4);
                        const __m512i x = _mm512_set1_epi32(v ^ (int)0x80808080);  // int8 code + 128, as unsigned
                        a0[j] = _mm512_dpbusd_epi32(a0[j], x, w0); a1[j] = _mm512_dpbusd_epi32(a1[j], x, w1);
                    }
                }
                const __m512 s0 = _mm512_loadu_ps(g->scale + ((size_t)(2 * rb) * g->ng + gi) * 16);
                const __m512 s1 = _mm512_loadu_ps(g->scale + ((size_t)(2 * rb + 1) * g->ng + gi) * 16);
                for (int j = 0; j < 8; j++) { f0[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0[j]), s0, f0[j]); f1[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1[j]), s1, f1[j]); }
            }
            const __m512 o0 = _mm512_loadu_ps(g->off + 32 * rb), o1 = _mm512_loadu_ps(g->off + 32 * rb + 16);
            const int r0 = 32 * rb, n0 = g->rows - r0 < 16 ? g->rows - r0 : 16, n1 = g->rows - r0 - 16 < 16 ? g->rows - r0 - 16 : 16;
            const __mmask16 m0 = n0 >= 16 ? 0xFFFF : (__mmask16)((1u << (n0 > 0 ? n0 : 0)) - 1);
            const __mmask16 m1 = n1 >= 16 ? 0xFFFF : (__mmask16)((1u << (n1 > 0 ? n1 : 0)) - 1);
            for (int j = 0; j < 8 && t0 + j < T; j++) {
                const __m512 st = _mm512_set1_ps(step[t0 + j]);
                float *o = OUTPTR(out, ldo, T, t0 + j, r0);
                __m512 v0 = _mm512_mul_ps(_mm512_sub_ps(f0[j], o0), st), v1 = _mm512_mul_ps(_mm512_sub_ps(f1[j], o1), st);
                if (acc) { v0 = _mm512_add_ps(v0, _mm512_maskz_loadu_ps(m0, o)); v1 = _mm512_add_ps(v1, _mm512_maskz_loadu_ps(m1, o + 16)); }
                _mm512_mask_storeu_ps(o, m0, v0); _mm512_mask_storeu_ps(o + 16, m1, v1);
            }
        }
}

// ---------------------------------------------------------------------------------- the same GEMM on AMX
// AMX (Sapphire Rapids and later): eight tile registers of 16 rows x 64 bytes. TDPBSSD multiplies a 16 x 64 int8 tile
// (16 tokens' activations, 64 inputs) by a 16 x 64 tile in VNNI order (the TG layout: 16 rows of 4-input groups for
// 16 weight rows) into 16 x 16 int32 sums: 16,384 multiply-adds per instruction. Signed x signed, so no offset. A
// block is 32 tokens x 32 rows (four sum tiles); after each 128-input scale group the sums are stored and scaled into
// float accumulators with AVX-512, which keeps the trained per-group scales exact.
#include <sys/syscall.h>
#include <unistd.h>
static int g_amx = 0;
typedef struct { unsigned char palette, start_row, reserved[14]; unsigned short colsb[16]; unsigned char rows[16]; } TileCfg;
static void amx_init(void) {
    if (getenv("GTSL_NOAMX")) return;
    if (syscall(SYS_arch_prctl, 0x1023 /* ARCH_REQ_XCOMP_PERM */, 18 /* XFEATURE_XTILEDATA */)) return;
    g_amx = 1;
}
#ifdef __AMX_INT8__
static void amx_config(void) {
    TileCfg c; memset(&c, 0, sizeof c);
    c.palette = 1;
    for (int i = 0; i < 8; i++) { c.rows[i] = 16; c.colsb[i] = 64; }
    _tile_loadconfig(&c);
}
static signed char *g_apack; static size_t g_apack_n;  // activations packed tile by tile
static void tg_gemm_amx(const TG *g, const signed char *xs_rows, int ldx, const float *step, int T, float *out, int ldo, int acc) {
    const int K4 = g->K / 4, nrb = g->rows_pad / 32, ntb = (T + 31) / 32, gsteps = g->gsz / 64, nk = g->K / 64;
    // pack: tile (16-token block tb16, 64-input block kb) is 1 KB contiguous, so every tile load is sequential
    const size_t need = (size_t)ntb * 2 * nk * 1024;
    if (need > g_apack_n) { free(g_apack); g_apack = (signed char *)xalloc(need); g_apack_n = need; }
    signed char *xs = g_apack;
#pragma omp parallel for collapse(2) schedule(static)
    for (int b16 = 0; b16 < ntb * 2; b16++)
        for (int kb = 0; kb < nk; kb++) {
            signed char *dst = xs + ((size_t)b16 * nk + kb) * 1024;
            for (int r = 0; r < 16; r++) {
                const int t = 16 * b16 + r;
                if (t < T) memcpy(dst + r * 64, xs_rows + (size_t)t * ldx + 64 * kb, 64); else memset(dst + r * 64, 0, 64);
            }
        }
#pragma omp parallel
    {
        amx_config();
        int sums[4][16][16] __attribute__((aligned(64)));
        float fa[32][32] __attribute__((aligned(64)));
#pragma omp for collapse(2) schedule(static)
        for (int rb = 0; rb < nrb; rb++)  // weight blocks outside: a 32-row block (32 KB) stays in cache for every token block
            for (int tb = 0; tb < ntb; tb++) {
                const int t0 = 32 * tb;
                const signed char *a0 = xs + (size_t)(2 * tb) * nk * 1024, *a1 = xs + (size_t)(2 * tb + 1) * nk * 1024;
                const signed char *w0 = g->w + (size_t)(2 * rb) * K4 * 64, *w1 = g->w + (size_t)(2 * rb + 1) * K4 * 64;
                memset(fa, 0, sizeof fa);
                for (int gi = 0; gi < g->ng; gi++) {
                    _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);
                    for (int s = 0; s < gsteps; s++) {
                        const int k0 = gi * g->gsz + 64 * s;
                        _tile_loadd(4, a0 + (size_t)(k0 / 64) * 1024, 64); _tile_loadd(5, a1 + (size_t)(k0 / 64) * 1024, 64);
                        _tile_loadd(6, w0 + (size_t)(k0 / 4) * 64, 64); _tile_loadd(7, w1 + (size_t)(k0 / 4) * 64, 64);
                        _tile_dpbssd(0, 4, 6); _tile_dpbssd(1, 4, 7); _tile_dpbssd(2, 5, 6); _tile_dpbssd(3, 5, 7);
                    }
                    _tile_stored(0, sums[0], 64); _tile_stored(1, sums[1], 64); _tile_stored(2, sums[2], 64); _tile_stored(3, sums[3], 64);
                    const __m512 s0 = _mm512_loadu_ps(g->scale + ((size_t)(2 * rb) * g->ng + gi) * 16);
                    const __m512 s1 = _mm512_loadu_ps(g->scale + ((size_t)(2 * rb + 1) * g->ng + gi) * 16);
                    for (int i = 0; i < 16; i++) {
                        __m512 *f = (__m512 *)fa[i], *f2 = (__m512 *)fa[16 + i];
                        f[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(sums[0][i])), s0, f[0]);
                        f[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(sums[1][i])), s1, f[1]);
                        f2[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(sums[2][i])), s0, f2[0]);
                        f2[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(sums[3][i])), s1, f2[1]);
                    }
                }
                const int r0 = 32 * rb, n0 = g->rows - r0 < 16 ? g->rows - r0 : 16, n1 = g->rows - r0 - 16 < 16 ? g->rows - r0 - 16 : 16;
                const __mmask16 m0 = n0 >= 16 ? 0xFFFF : (__mmask16)((1u << (n0 > 0 ? n0 : 0)) - 1);
                const __mmask16 m1 = n1 >= 16 ? 0xFFFF : (__mmask16)((1u << (n1 > 0 ? n1 : 0)) - 1);
                for (int j = 0; j < 32 && t0 + j < T; j++) {
                    const __m512 st = _mm512_set1_ps(step[t0 + j]);
                    float *o = OUTPTR(out, ldo, T, t0 + j, r0);
                    __m512 v0 = _mm512_mul_ps(_mm512_load_ps(fa[j]), st), v1 = _mm512_mul_ps(_mm512_load_ps(fa[j] + 16), st);
                    if (acc) { v0 = _mm512_add_ps(v0, _mm512_maskz_loadu_ps(m0, o)); v1 = _mm512_add_ps(v1, _mm512_maskz_loadu_ps(m1, o + 16)); }
                    _mm512_mask_storeu_ps(o, m0, v0); _mm512_mask_storeu_ps(o + 16, m1, v1);
                }
            }
        _tile_release();
    }
}
#endif

#ifdef __AMX_BF16__
// ------------------------------------------------------------------------ chunked scans on AMX (bf16)
// Over a chunk of 64 tokens (i, j) in scan order, one head's two-way recurrence is four 64 x 64 x 64 products:
//   G = (C B^T) * exp(c_i - c_j) * dt_j, kept for j <= i (forward, own term included) or j < i (backward, excluded)
//   Y = G X + (C * exp(c_i)) S            (S: the state carried in from earlier chunks; c: running sum of dt * A)
//   S = exp(c_last) S + (B * exp(c_last - c_j) dt_j)^T X
// every exponent is <= 0. Inputs are rounded to bf16, sums are float32 (as the GPU's chunked scan).
static inline unsigned short bf16(float f) { unsigned u; memcpy(&u, &f, 4); return (unsigned short)((u + 0x7FFF + ((u >> 16) & 1)) >> 16); }
// C (64 x 64 float, row-major) = or += A (64 x 64 bf16, row-major [i][k]) @ B (bf16 in pairs: Bv[k / 2][n][k % 2])
static void mm64(const unsigned short *A, const unsigned short *Bv, float *C, int acc) {
    for (int bi = 0; bi < 2; bi++)
        for (int bj = 0; bj < 2; bj++) {
            float *c0 = C + (32 * bi) * 64 + 32 * bj;
            if (acc) { _tile_loadd(0, c0, 256); _tile_loadd(1, c0 + 16, 256); _tile_loadd(2, c0 + 16 * 64, 256); _tile_loadd(3, c0 + 16 * 64 + 16, 256); }
            else { _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3); }
            for (int ks = 0; ks < 2; ks++) {
                _tile_loadd(4, A + (32 * bi) * 64 + 32 * ks, 128); _tile_loadd(5, A + (32 * bi + 16) * 64 + 32 * ks, 128);
                _tile_loadd(6, Bv + (16 * ks) * 128 + (32 * bj) * 2, 256); _tile_loadd(7, Bv + (16 * ks) * 128 + (32 * bj + 16) * 2, 256);
                _tile_dpbf16ps(0, 4, 6); _tile_dpbf16ps(1, 4, 7); _tile_dpbf16ps(2, 5, 6); _tile_dpbf16ps(3, 5, 7);
            }
            _tile_stored(0, c0, 256); _tile_stored(1, c0 + 16, 256); _tile_stored(2, c0 + 16 * 64, 256); _tile_stored(3, c0 + 16 * 64 + 16, 256);
        }
}
#endif

static void tg_gemm(const TG *g, const signed char *xs, int ldx, const float *step, int T, float *out, int ldo, int acc) {
#ifdef __AMX_INT8__
    if (g_amx && g->gsz % 64 == 0) { tg_gemm_amx(g, xs, ldx, step, T, out, ldo, acc); return; }
#endif
    tg_gemm_vnni(g, xs, ldx, step, T, out, ldo, acc);
}

static void to_u8(const signed char *q, signed char *u, int n) { memcpy(u, q, n); }  // the GEMMs' activation copy (contiguous)

typedef struct {
    int has_norm1;
    float *norm1, *norm2;
    TMat in, out, node_in, node_out, ldown, lup;
    TG gin, gout, gdown, gup;
    float *dt_w, *dt_b, *A, *Dsk, *node_bias, *lbias;
} Layer;

static int g_gemm = 1;  // 0: per-token row dot products (the first kernel), for comparison
static int V, D, NL, H, N, P, NT, DEPTH, R, PER, NN, ZR;  // ZR: in_proj rows (2HN + D)
static float EPS;
static float *emb, *emb_norm, *norm_f, *head_dense, *head_norm, *dec_bias;
static Layer *layers;

static void pack_or_die(TMat *m, float *w, int rows, int cols) {
    if (!tm_pack(m, w, rows, cols)) { fprintf(stderr, "a ternary tensor (%d x %d) did not pack\n", rows, cols); exit(1); }
    free(w);
}

static void layernorm(const float *x, const float *w, float *y, int n) {
    float mu = 0, var = 0;
    for (int i = 0; i < n; i++) mu += x[i];
    mu /= n;
    for (int i = 0; i < n; i++) { const float c = x[i] - mu; var += c * c; }
    const float s = 1.0f / sqrtf(var / n + EPS);
    for (int i = 0; i < n; i++) y[i] = (x[i] - mu) * s * w[i];
}

// ----------------------------------------------------------------------------------------------- scratch
static float *X, *U, *Z, *DT, *YF, *YB, *OUT, *STEP, *LZ;
static signed char *Q, *LQ;
static signed char *QU, *LQU;  // contiguous int8 codes for the GEMMs, padded by 32 tokens
static double t_proj_in, t_scan, t_proj_out, t_trees, t_lin, t_sub[5];  // sub: LN+quant, dt, in GEMM, combine+quant, out GEMM

static void alloc_scratch(int T) {
    X = (float *)xalloc((size_t)T * D * 4); U = (float *)xalloc((size_t)T * D * 4); OUT = (float *)xalloc((size_t)T * D * 4);
    Z = (float *)xalloc((size_t)T * ZR * 4); DT = (float *)xalloc((size_t)T * H * 4);
    YF = (float *)xalloc((size_t)T * D * 4); YB = (float *)xalloc((size_t)T * D * 4);
    STEP = (float *)xalloc((size_t)T * 4); Q = (signed char *)xalloc((size_t)T * (D + 64));
    QU = (signed char *)xalloc((size_t)(T + 32) * D);
    if (R) { LZ = (float *)xalloc((size_t)T * R * 4); LQ = (signed char *)xalloc((size_t)T * (R + 64)); LQU = (signed char *)xalloc((size_t)(T + 32) * R); }
}

static inline void load_q(const signed char *q, int n, __m512i *xq) { for (int c = 0; c < n / 64; c++) xq[c] = _mm512_loadu_si512(q + 64 * c); }

// --------------------------------------------------------------------------------------------- the BiSSD
#ifdef __AMX_BF16__
// vectorised operand preparation: bf16 rows, pair-interleaved rows (k, k + 1 into one 32-bit word), transposes
static inline void cvt_bf16(const float *x, unsigned short *y, int n) {
    for (int i = 0; i < n; i += 16) _mm256_storeu_si256((__m256i *)(y + i), (__m256i)_mm512_cvtneps_pbh(_mm512_loadu_ps(x + i)));
}
static inline void pair_rows(const unsigned short *r, unsigned *w) {  // r: 64 x 64 bf16 [k][p] -> w[k / 2][p]
    for (int kp = 0; kp < 32; kp++)
        for (int p = 0; p < 64; p += 16) {
            const __m512i lo = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(r + (2 * kp) * 64 + p)));
            const __m512i hi = _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(r + (2 * kp + 1) * 64 + p)));
            _mm512_storeu_si512(w + kp * 64 + p, _mm512_or_si512(lo, _mm512_slli_epi32(hi, 16)));
        }
}
static inline void tr16(__m512i r[16]) {  // 16 x 16 32-bit transpose in registers
    __m512i t[16];
    for (int i = 0; i < 8; i++) { t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]); t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]); }
    for (int i = 0; i < 4; i++) {
        r[4 * i] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]); r[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        r[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]); r[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) { t[8 * i + j] = _mm512_shuffle_i32x4(r[8 * i + j], r[8 * i + 4 + j], 0x88); t[8 * i + 4 + j] = _mm512_shuffle_i32x4(r[8 * i + j], r[8 * i + 4 + j], 0xdd); }
    for (int j = 0; j < 8; j++) { r[j] = _mm512_shuffle_i32x4(t[j], t[8 + j], 0x88); r[8 + j] = _mm512_shuffle_i32x4(t[j], t[8 + j], 0xdd); }
}
static inline void transpose_words(const unsigned *src, int rows, int cols, unsigned *dst) {  // [rows][cols] -> [cols][rows]
    for (int rb = 0; rb < rows; rb += 16)
        for (int cb = 0; cb < cols; cb += 16) {
            __m512i r[16];
            for (int i = 0; i < 16; i++) r[i] = _mm512_loadu_si512(src + (size_t)(rb + i) * cols + cb);
            tr16(r);
            for (int k = 0; k < 16; k++) _mm512_storeu_si512(dst + (size_t)(cb + k) * rows + rb, r[k]);
        }
}
static void scans_amx(const Layer *L, int T) {
#pragma omp parallel
    {
        amx_config();
        static __thread float S[64 * 64], G[64 * 64], Y[64 * 64], Cc[64 * 64], Bc[64 * 64], Xc[64 * 64], Ft[64 * 64];
        static __thread unsigned short Ab[64 * 64], Tb[64 * 64], Bv[64 * 64], Xv[64 * 64], Sv[64 * 64];
        static __thread float c[64], dts[64], e[64], w[64];
#pragma omp for schedule(dynamic, 1)
        for (int job = 0; job < 2 * H; job++) {
            const int h = job / 2, dir = job % 2;
            float *Yout = dir ? YB : YF;
            memset(S, 0, sizeof S);
            for (int c0 = 0; c0 < T; c0 += 64) {
                const int n = T - c0 < 64 ? T - c0 : 64;
                int idx[64];
                float run = 0;
                for (int i = 0; i < 64; i++) {
                    idx[i] = i < n ? (dir ? T - 1 - (c0 + i) : c0 + i) : -1;
                    if (i < n) {  // the head's panels: C, B, x of token t at Z + (panel * T + t) * 64
                        const size_t t = idx[i];
                        memcpy(Cc + i * 64, Z + ((size_t)h * T + t) * 64, 256); memcpy(Bc + i * 64, Z + ((size_t)(H + h) * T + t) * 64, 256);
                        memcpy(Xc + i * 64, Z + ((size_t)(2 * H + h) * T + t) * 64, 256);
                        dts[i] = DT[t * H + h];
                    } else {
                        memset(Cc + i * 64, 0, 256); memset(Bc + i * 64, 0, 256); memset(Xc + i * 64, 0, 256); dts[i] = 0;
                    }
                    run += dts[i] * L->A[h];
                    c[i] = run;
                }
                const float clast = c[63];
                for (int i = 0; i < 64; i++) { e[i] = expf(c[i]); w[i] = expf(clast - c[i]) * dts[i]; }
                // G = C B^T
                cvt_bf16(Cc, Ab, 4096);
                cvt_bf16(Bc, Tb, 4096);
                transpose_words((const unsigned *)Tb, 64, 32, (unsigned *)Bv);  // B^T in pairs: word [n / 2][j] = B[j][n, n + 1]
                mm64(Ab, Bv, G, 0);
                // decay, dt and the causal mask, row by row
                if (clast > -60.0f) {  // exp(c_i - c_j) = exp(c_i) exp(-c_j): 128 exponentials instead of 4,096
                    float einv[64];
                    for (int j = 0; j < 64; j++) einv[j] = dts[j] / e[j];
                    for (int i = 0; i < 64; i++) {
                        float *g = G + i * 64;
                        const int lim = dir ? i : i + 1;
                        const float ei = e[i];
                        for (int j = 0; j < 64; j++) g[j] = j < lim ? g[j] * (ei * einv[j]) : 0.0f;
                    }
                } else {  // a chunk with strong decay: each factor on its own, so nothing overflows
                    for (int i = 0; i < 64; i++) {
                        float *g = G + i * 64, ex[64];
                        const int lim = dir ? i : i + 1;
                        for (int j = 0; j < 64; j++) ex[j] = c[i] - c[j];
                        for (int j = 0; j < 64; j++) ex[j] = expf(ex[j] < 0 ? ex[j] : 0);
                        for (int j = 0; j < 64; j++) g[j] = j < lim ? g[j] * ex[j] * dts[j] : 0.0f;
                    }
                }
                cvt_bf16(G, Ab, 4096);
                cvt_bf16(Xc, Tb, 4096);
                pair_rows(Tb, (unsigned *)Xv);
                mm64(Ab, Xv, Y, 0);
                // + (C exp(c_i)) S
                for (int i = 0; i < 64; i++) for (int k = 0; k < 64; k++) Ft[i * 64 + k] = Cc[i * 64 + k] * e[i];
                cvt_bf16(Ft, Ab, 4096);
                cvt_bf16(S, Tb, 4096);
                pair_rows(Tb, (unsigned *)Sv);
                mm64(Ab, Sv, Y, 1);
                // S = exp(c_last) S + (B w)^T X
                const float el = expf(clast);
                for (int k = 0; k < 4096; k++) S[k] *= el;
                for (int j = 0; j < 64; j++) for (int k = 0; k < 64; k++) Ft[j * 64 + k] = Bc[j * 64 + k] * w[j];
                cvt_bf16(Ft, Tb, 4096);  // (B w) [j][n] in bf16; transposed to [n][j] as pairs over j, then words
                pair_rows(Tb, (unsigned *)Sv);
                transpose_words((const unsigned *)Sv, 32, 64, (unsigned *)Ab);
                mm64(Ab, Xv, S, 1);
                for (int i = 0; i < n; i++) memcpy(Yout + ((size_t)h * T + idx[i]) * 64, Y + i * 64, 256);  // panels
            }
        }
        _tile_release();
    }
}
#endif
static int bissd_scans_only = 0;
static void scan_recurrence(const Layer *L, int T) {
#pragma omp parallel for schedule(dynamic, 1)
    for (int job = 0; job < 2 * H; job++) {
        const int h = job / 2, dir = job % 2, PV = P / 16;
        __m512 S[64][4];
        for (int n = 0; n < N; n++) for (int v = 0; v < PV; v++) S[n][v] = _mm512_setzero_ps();
        float *Y = dir ? YB : YF;
        // Two tokens per sweep over the state: each row of S is loaded once, updated by token t, read, updated by the
        // next token, read, and stored once (S does not fit in registers, so its L1 traffic is the cost).
        for (int k = 0; k < T; k += 2) {
            const int two = k + 1 < T, t0 = dir ? T - 1 - k : k, t1 = dir ? t0 - 1 : t0 + 1;
            const int tt[2] = {t0, two ? t1 : t0};
            const float *C[2], *B[2], *x[2];
            float dt[2];
            __m512 a[2], xv[2][4], y[2][4];
            for (int j = 0; j < 2; j++) {
                const float *z = Z + (size_t)tt[j] * ZR;
                C[j] = z + h * N; B[j] = z + H * N + h * N; x[j] = z + 2 * H * N + h * P;
                dt[j] = DT[(size_t)tt[j] * H + h];
                a[j] = _mm512_set1_ps(expf(dt[j] * L->A[h]));
                for (int v = 0; v < PV; v++) { xv[j][v] = _mm512_loadu_ps(x[j] + 16 * v); y[j][v] = _mm512_setzero_ps(); }
            }
            for (int n = 0; n < N; n++) {
                __m512 s0 = S[n][0], s1 = S[n][1], s2 = S[n][2], s3 = S[n][3];
                for (int j = 0; j < 1 + two; j++) {
                    const __m512 b = _mm512_set1_ps(dt[j] * B[j][n]), c = _mm512_set1_ps(C[j][n]);
                    if (!dir) {  // decay, write, read
                        s0 = _mm512_fmadd_ps(s0, a[j], _mm512_mul_ps(b, xv[j][0])); s1 = _mm512_fmadd_ps(s1, a[j], _mm512_mul_ps(b, xv[j][1]));
                        s2 = _mm512_fmadd_ps(s2, a[j], _mm512_mul_ps(b, xv[j][2])); s3 = _mm512_fmadd_ps(s3, a[j], _mm512_mul_ps(b, xv[j][3]));
                        y[j][0] = _mm512_fmadd_ps(c, s0, y[j][0]); y[j][1] = _mm512_fmadd_ps(c, s1, y[j][1]);
                        y[j][2] = _mm512_fmadd_ps(c, s2, y[j][2]); y[j][3] = _mm512_fmadd_ps(c, s3, y[j][3]);
                    } else {  // decay, read, write
                        s0 = _mm512_mul_ps(s0, a[j]); s1 = _mm512_mul_ps(s1, a[j]); s2 = _mm512_mul_ps(s2, a[j]); s3 = _mm512_mul_ps(s3, a[j]);
                        y[j][0] = _mm512_fmadd_ps(c, s0, y[j][0]); y[j][1] = _mm512_fmadd_ps(c, s1, y[j][1]);
                        y[j][2] = _mm512_fmadd_ps(c, s2, y[j][2]); y[j][3] = _mm512_fmadd_ps(c, s3, y[j][3]);
                        s0 = _mm512_fmadd_ps(b, xv[j][0], s0); s1 = _mm512_fmadd_ps(b, xv[j][1], s1);
                        s2 = _mm512_fmadd_ps(b, xv[j][2], s2); s3 = _mm512_fmadd_ps(b, xv[j][3], s3);
                    }
                }
                S[n][0] = s0; S[n][1] = s1; S[n][2] = s2; S[n][3] = s3;
            }
            for (int j = 0; j < 1 + two; j++)
                for (int v = 0; v < PV; v++) _mm512_storeu_ps(Y + (size_t)tt[j] * D + h * P + 16 * v, y[j][v]);
        }
    }
}

static void bissd(const Layer *L, int T) {
    double t0 = now();
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {  // LN1, int8, in_proj (C, B, x), dt
        float *u = U + (size_t)t * D;
        if (L->has_norm1) layernorm(X + (size_t)t * D, L->norm1, u, D); else memcpy(u, X + (size_t)t * D, D * 4);
        signed char *q = Q + (size_t)t * (D + 64);
        const float st = STEP[t] = quant8(u, q, D);
        if (g_gemm) to_u8(q, QU + (size_t)t * D, D);
        else {
            __m512i xq[16];
            load_q(q, D, xq);
            float *z = Z + (size_t)t * ZR;
            for (int r = 0; r < ZR; r++) z[r] = st * tm_dot_i8(&L->in, r, xq);
        }
    }
    double ta = now(); t_sub[0] += ta - t0;
    // dt = softplus(dt_proj u), in float, 16 tokens at a time so each weight row is read once per block
#pragma omp parallel for schedule(static)
    for (int t0 = 0; t0 < T; t0 += 4) {  // 4 tokens x 4 rows of 16 accumulators each: 16 independent FMA chains
        const int nb = T - t0 < 4 ? T - t0 : 4;
        const float *u[4];
        for (int j = 0; j < 4; j++) u[j] = U + (size_t)(t0 + (j < nb ? j : 0)) * D;
        float pre[4][64], tmp[64];
        for (int h = 0; h < H; h += 4) {
            __m512 acc[4][4];
            for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) acc[j][i] = _mm512_setzero_ps();
            const float *w0 = L->dt_w + (size_t)h * D;
            for (int k = 0; k < D; k += 16) {
                __m512 wv[4];
                for (int i = 0; i < 4; i++) wv[i] = _mm512_loadu_ps(w0 + (size_t)i * D + k);
                for (int j = 0; j < 4; j++) {
                    const __m512 uv = _mm512_loadu_ps(u[j] + k);
                    for (int i = 0; i < 4; i++) acc[j][i] = _mm512_fmadd_ps(uv, wv[i], acc[j][i]);
                }
            }
            for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) pre[j][h + i] = _mm512_reduce_add_ps(acc[j][i]) + L->dt_b[h + i];
        }
        for (int j = 0; j < nb; j++) softplus_array(pre[j], DT + (size_t)(t0 + j) * H, tmp, H);
    }
    double tb = now(); t_sub[1] += tb - ta;
    const int panel = g_gemm && g_amx && N == 64 && P == 64 && !getenv("GTSL_RECUR");  // chunked scans read panels
    if (g_gemm) tg_gemm(&L->gin, QU, D, STEP, T, Z, panel ? 0 : ZR, 0);
    double t1 = now(); t_sub[2] += t1 - tb;
    t_proj_in += t1 - t0;
    // the scans: one (head, direction) per job; the head's state S is N rows of P floats
#ifdef __AMX_BF16__
    if (panel) { scans_amx(L, T); goto scanned; }
#endif
    scan_recurrence(L, T);
    goto scanned;
scanned:;
    double t2 = now();
    t_scan += t2 - t1;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {  // y = forward + backward + D x; int8; out_proj; residual
        float y[1024];
        if (panel)
            for (int h = 0; h < H; h++) {
                const float *f = YF + ((size_t)h * T + t) * 64, *b = YB + ((size_t)h * T + t) * 64, *x = Z + ((size_t)(2 * H + h) * T + t) * 64;
                const float dk = L->Dsk[h];
                for (int p = 0; p < 64; p++) y[h * 64 + p] = f[p] + b[p] + dk * x[p];
            }
        else {
            const float *x = Z + (size_t)t * ZR + 2 * H * N;
            for (int c = 0; c < D; c++) y[c] = YF[(size_t)t * D + c] + YB[(size_t)t * D + c] + L->Dsk[c / P] * x[c];
        }
        signed char *q = Q + (size_t)t * (D + 64);
        const float st = STEP[t] = quant8(y, q, D);
        if (g_gemm) { to_u8(q, QU + (size_t)t * D, D); continue; }
        __m512i xq[16];
        load_q(q, D, xq);
        float *h = X + (size_t)t * D;
        for (int r = 0; r < D; r++) h[r] += st * tm_dot_i8(&L->out, r, xq);
    }
    double tc = now(); t_sub[3] += tc - t2;
    if (g_gemm) tg_gemm(&L->gout, QU, D, STEP, T, X, D, 1);
    t_sub[4] += now() - tc;
    t_proj_out += now() - t2;
}

// ------------------------------------------------------------------------- deep trees and the linear path
static inline void prefetch_row(const TMat *m, int r) {  // a packed row: two bit masks and its scales
    const char *p = (const char *)(m->pos + (size_t)r * m->nch), *n = (const char *)(m->neg + (size_t)r * m->nch);
    for (int b = 0; b < m->nch * 2; b += 64) { _mm_prefetch(p + b, _MM_HINT_T0); _mm_prefetch(n + b, _MM_HINT_T0); }
    _mm_prefetch((const char *)(m->scale + (size_t)r * m->ng), _MM_HINT_T0);
}

static void ffn(const Layer *L, int T) {
    double t0 = now(), tl = 0;
    // Trees: groups of 4 tokens walk their 4 trees in lockstep, and each next node's rows are prefetched as soon as
    // the branch is known, so up to 16 independent row fetches overlap (the node tables do not fit in cache).
#pragma omp parallel for schedule(static) reduction(+ : tl)
    for (int g0 = 0; g0 < T; g0 += 4) {
        const int ng = T - g0 < 4 ? T - g0 : 4;
        __m512i xq[4][16];
        float st[4];
        int rows[4][256], cur[4][16];
        float lg[4][256], coef[256];
        for (int g = 0; g < ng; g++) {
            const int t = g0 + g;
            float *u = U + (size_t)t * D;
            layernorm(X + (size_t)t * D, L->norm2, u, D);
            signed char *q = Q + (size_t)t * (D + 64);
            st[g] = STEP[t] = quant8(u, q, D);
            load_q(q, D, xq[g]);
            for (int tr = 0; tr < NT; tr++) cur[g][tr] = 0;
            if (R && g_gemm) to_u8(q, QU + (size_t)t * D, D);
        }
        int S = 0;
        for (int k = 0; k <= DEPTH; k++, S += NT)
            for (int g = 0; g < ng; g++)
                for (int tr = 0; tr < NT; tr++) {
                    const int gid = tr * PER + cur[g][tr];
                    const float l = st[g] * tm_dot_i8(&L->node_in, gid, xq[g]) + L->node_bias[gid];
                    rows[g][S + tr] = gid; lg[g][S + tr] = l;
                    prefetch_row(&L->node_out, gid);
                    cur[g][tr] = 2 * cur[g][tr] + 1 + (l > 0);
                    if (k < DEPTH) prefetch_row(&L->node_in, tr * PER + cur[g][tr]);
                }
        for (int g = 0; g < ng; g++) {
            const int t = g0 + g;
            float *o = OUT + (size_t)t * D;
            gelu_array(lg[g], coef, S);
            memset(o, 0, D * 4);
            tm_out_rows(&L->node_out, rows[g], coef, S, o);
            if (R && !g_gemm) {
                const double s0 = now();
                float *lz = LZ + (size_t)t * R;
                for (int r = 0; r < R; r++) lz[r] = st[g] * tm_dot_i8(&L->ldown, r, xq[g]);
                signed char *lq = LQ + (size_t)t * (R + 64);
                const float s2 = quant8(lz, lq, R);
                __m512i zq[16];
                load_q(lq, R, zq);
                for (int c = 0; c < D; c++) o[c] += s2 * tm_dot_i8(&L->lup, c, zq) + L->lbias[c];
                tl += now() - s0;
            }
            if (!(R && g_gemm)) { float *h = X + (size_t)t * D; for (int c = 0; c < D; c++) h[c] += o[c]; }
        }
    }
    if (R && g_gemm) {  // the linear path over all tokens: down, int8, up into the trees' output, then the residual
        const double s0 = now();
        tg_gemm(&L->gdown, QU, D, STEP, T, LZ, R, 0);
        float *st2 = DT;  // a free T-float buffer here (dt is only used inside the BiSSD)
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) { st2[t] = quant8(LZ + (size_t)t * R, LQ + (size_t)t * (R + 64), R); to_u8(LQ + (size_t)t * (R + 64), LQU + (size_t)t * R, R); }
        tg_gemm(&L->gup, LQU, R, st2, T, OUT, D, 1);
#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; t++) { float *h = X + (size_t)t * D, *o = OUT + (size_t)t * D; for (int c = 0; c < D; c++) h[c] += o[c] + L->lbias[c]; }
        int th = 1;
#ifdef _OPENMP
        th = omp_get_max_threads();
#endif
        tl += (now() - s0) * th;
    }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    t_lin += tl / threads;
    t_trees += now() - t0 - tl / threads;
}

static void encode(const int *ids, int T, float *hidden) {
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) layernorm(emb + (size_t)ids[t] * D, emb_norm, X + (size_t)t * D, D);
    for (int l = 0; l < NL; l++) { bissd(&layers[l], T); ffn(&layers[l], T); }
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) layernorm(X + (size_t)t * D, norm_f, hidden + (size_t)t * D, D);
}

static void head(const float *h, float *logits) {  // dense (float), gelu, LN, tied decoder
    float z[1024], g[1024], n[1024];
#pragma omp parallel for schedule(static)
    for (int r = 0; r < D; r++) z[r] = dot(head_dense + (size_t)r * D, h, D);
    gelu_array(z, g, D);
    layernorm(g, head_norm, n, D);
#pragma omp parallel for schedule(static)
    for (int v = 0; v < V; v++) logits[v] = dot(emb + (size_t)v * D, n, D) + dec_bias[v];
}

// ------------------------------------------------------------------------------------------------ loading
static void load_layer(Layer *L) {
    L->has_norm1 = rdi();
    if (L->has_norm1) L->norm1 = rdf(D);
    L->norm2 = rdf(D);
    pack_or_die(&L->in, rdf((size_t)ZR * D), ZR, D);
    L->dt_w = rdf((size_t)H * D); L->dt_b = rdf(H); L->A = rdf(H); L->Dsk = rdf(H);
    for (int h = 0; h < H; h++) L->A[h] = -expf(L->A[h]);
    pack_or_die(&L->out, rdf((size_t)D * D), D, D);
    pack_or_die(&L->node_in, rdf((size_t)NN * D), NN, D);
    L->node_bias = rdf(NN);
    pack_or_die(&L->node_out, rdf((size_t)NN * D), NN, D);
    if (R) { pack_or_die(&L->ldown, rdf((size_t)R * D), R, D); pack_or_die(&L->lup, rdf((size_t)D * R), D, R); L->lbias = rdf(D); }
}

static float *frnd(size_t n, float k) { float *p = (float *)xalloc(n * 4); for (size_t i = 0; i < n; i++) p[i] = (2 * urand() - 1) * k; return p; }
static int g_rowscale = 0;  // synth: the dense projections with one scale per row
static void tm_rowscale(TMat *m) { for (int r = 0; r < m->rows; r++) for (int gi = 1; gi < m->ng; gi++) m->scale[(size_t)r * m->ng + gi] = m->scale[(size_t)r * m->ng]; }
static void synth_layer(Layer *L, int idx) {
    L->has_norm1 = idx > 0;
    L->norm1 = frnd(D, 0); for (int c = 0; c < D; c++) L->norm1[c] = 1;
    L->norm2 = L->norm1;
    tm_random(&L->in, ZR, D); tm_random(&L->out, D, D);
    L->dt_w = frnd((size_t)H * D, 0.01f); L->dt_b = frnd(H, 0.1f); L->A = frnd(H, 0.0f); L->Dsk = frnd(H, 1.0f);
    for (int h = 0; h < H; h++) L->A[h] = -0.2f;
    tm_random(&L->node_in, NN, D); tm_random(&L->node_out, NN, D); L->node_bias = frnd(NN, 0.1f);
    if (R) { tm_random(&L->ldown, R, D); tm_random(&L->lup, D, R); L->lbias = frnd(D, 0.1f); }
    if (g_rowscale) { tm_rowscale(&L->in); tm_rowscale(&L->out); if (R) { tm_rowscale(&L->ldown); tm_rowscale(&L->lup); } }
}

static void build_gemm(Layer *L) {
    tg_from_tm(&L->gin, &L->in); tg_from_tm(&L->gout, &L->out);
    if (R) { tg_from_tm(&L->gdown, &L->ldown); tg_from_tm(&L->gup, &L->lup); }
}

static void set_shape(void) { PER = (1 << (DEPTH + 1)) - 1; NN = NT * PER; ZR = 2 * H * N + D; }

static void time_it(const int *ids, int T, int reps, float *hidden) {
    int *seq = (int *)xalloc(512 * 4);
    for (int t = 0; t < 512; t++) seq[t] = ids[t % T];
    const int lens[2] = {128, 512};
    for (int li = 0; li < 2; li++) {
        const int L = lens[li];
        encode(seq, L, hidden);  // warm
        double best = 1e30;
        t_proj_in = t_scan = t_proj_out = t_trees = t_lin = 0; memset(t_sub, 0, sizeof t_sub);
        for (int r = 0; r < reps; r++) { const double t0 = now(); encode(seq, L, hidden); const double dt = now() - t0; if (dt < best) best = dt; }
        const double tot = t_proj_in + t_scan + t_proj_out + t_trees + t_lin;
        printf("  sequence %3d: %.2f ms (best of %d) = %.1f us per token = %.0f tokens/s.  Share: in_proj %.0f%%, scans %.0f%%, "
               "out_proj %.0f%%, trees %.0f%%, linear path %.0f%%\n", L, best * 1e3, reps, best / L * 1e6, L / best,
               100 * t_proj_in / tot, 100 * t_scan / tot, 100 * t_proj_out / tot, 100 * t_trees / tot, 100 * t_lin / tot);
        printf("    within: LN1 + int8 %.0f%%, dt %.0f%%, in GEMM %.0f%%, combine + int8 %.0f%%, out GEMM %.0f%%\n",
               100 * t_sub[0] / tot, 100 * t_sub[1] / tot, 100 * t_sub[2] / tot, 100 * t_sub[3] / tot, 100 * t_sub[4] / tot);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gtsl_bench gtsl.bin [repeats] | gtsl_bench synth [layers] [linear_rank]\n"); return 1; }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (getenv("GTSL_ROWDOT")) g_gemm = 0;
    if (getenv("GTSL_ROWSCALE")) g_rowscale = 1;
    amx_init();
    if (!strcmp(argv[1], "synth")) {
        V = 50368; D = 1024; NL = argc > 2 ? atoi(argv[2]) : 28; H = 16; N = getenv("GTSL_SYNTH_N") ? atoi(getenv("GTSL_SYNTH_N")) : 64; P = 64; NT = 4; DEPTH = 9;
        R = argc > 3 ? atoi(argv[3]) : 0; EPS = 1e-5f;
        set_shape();
        emb = frnd((size_t)V * D, 0.05f); emb_norm = frnd(D, 0); norm_f = emb_norm;
        for (int c = 0; c < D; c++) emb_norm[c] = 1;
        layers = (Layer *)xalloc(NL * sizeof(Layer));
        for (int l = 0; l < NL; l++) { synth_layer(&layers[l], l); build_gemm(&layers[l]); }
        alloc_scratch(512);
        printf("%s. ", g_amx ? "AMX int8" : "AVX-512 VNNI");
        printf("%d thread(s). Synthetic GTS-L: %d layers, width %d, BiSSD %d heads x (state %d, head %d), %d trees of depth %d, linear rank %d\n",
               threads, NL, D, H, N, P, NT, DEPTH, R);
        int ids[512];
        for (int t = 0; t < 512; t++) ids[t] = 1000 + t;
        float *hidden = (float *)xalloc((size_t)512 * D * 4);
        if (getenv("GTSL_GEMMBENCH")) {  // in_proj's GEMM alone, 512 tokens
            for (size_t i = 0; i < (size_t)512 * D; i++) QU[i] = (signed char)((int)(rnd64() % 255) - 127);
            for (int t = 0; t < 512; t++) STEP[t] = 0.01f;
            tg_gemm(&layers[0].gin, QU, D, STEP, 512, Z, ZR, 0);
            const double t0 = now();
            for (int r = 0; r < 20; r++) tg_gemm(&layers[0].gin, QU, D, STEP, 512, Z, ZR, 0);
            const double dt = (now() - t0) / 20;
            printf("  in_proj GEMM, 512 tokens: %.3f ms = %.1f GMAC/s (%.1f us per token)\n", dt * 1e3, 512.0 * ZR * D / dt / 1e9, dt / 512 * 1e6);
            return 0;
        }
        time_it(ids, 512, 3, hidden);
        return 0;
    }
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    g_f = fopen(argv[1], "rb"); if (!g_f) { perror("open"); return 1; }
    if (rdi() != 11) { fprintf(stderr, "not a GTS-L file\n"); return 1; }
    V = rdi(); D = rdi(); NL = rdi(); H = rdi(); N = rdi(); P = rdi(); NT = rdi(); DEPTH = rdi(); R = rdi();
    if (fread(&EPS, 4, 1, g_f) != 1) return 1;
    set_shape();
    if (D % 128 || D > 1024 || N > 64 || P != 64 || H > 64 || H % 4 || H * P != D || NT > 16 || (R && (R % 64 || R > 1024))) { fprintf(stderr, "unsupported shape\n"); return 1; }
    emb = rdf((size_t)V * D); emb_norm = rdf(D); norm_f = rdf(D); head_dense = rdf((size_t)D * D); head_norm = rdf(D); dec_bias = rdf(V);
    layers = (Layer *)xalloc(NL * sizeof(Layer));
    for (int l = 0; l < NL; l++) { load_layer(&layers[l]); build_gemm(&layers[l]); }
    const int T = rdi(), M = rdi();
    int *ids = (int *)xalloc(T * 4), *pos = (int *)xalloc(M * 4);
    if (fread(ids, 4, T, g_f) != (size_t)T || fread(pos, 4, M, g_f) != (size_t)M) return 1;
    float *ref = rdf((size_t)M * V);
    fclose(g_f);
    alloc_scratch(512 > T ? 512 : T);
    printf("%s. ", g_amx ? "AMX int8" : "AVX-512 VNNI");
    printf("%d thread(s). GTS-L: %d layers, width %d, BiSSD %d heads x (state %d, head %d), %d trees of depth %d, linear rank %d; %d tensors packed ternary\n",
           threads, NL, D, H, N, P, NT, DEPTH, R, g_packed);
    float *hidden = (float *)xalloc((size_t)(512 > T ? 512 : T) * D * 4), *logits = (float *)xalloc((size_t)V * 4);
    encode(ids, T, hidden);
    float worst = 0; int agree = 0; double nll_c = 0, nll_r = 0;
    for (int k = 0; k < M; k++) {
        head(hidden + (size_t)pos[k] * D, logits);
        const float *r = ref + (size_t)k * V;
        int a = 0, b = 0;
        for (int v = 0; v < V; v++) { const float e = fabsf(logits[v] - r[v]); if (e > worst) worst = e; if (logits[v] > logits[a]) a = v; if (r[v] > r[b]) b = v; }
        agree += a == b;
        double zc = 0, zr = 0;
        for (int v = 0; v < V; v++) { zc += exp(logits[v] - logits[a]); zr += exp(r[v] - r[b]); }
        nll_c += log(zc) + logits[a] - logits[b]; nll_r += log(zr);
    }
    printf("  check on %d tokens, %d masked: max |logit - PyTorch| = %.2e, same top-1 on %d/%d; -log p(PyTorch's top-1): kernel %.4f, PyTorch %.4f\n",
           T, M, worst, agree, M, nll_c / M, nll_r / M);
    time_it(ids, T, reps, hidden);
    return 0;
}
