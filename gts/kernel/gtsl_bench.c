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

// ------------------------------------------------------------------------------- ternary GEMM (VNNI)
// The dense projections run over all tokens at once. Weights as int8 codes {-1, 0, +1}, laid out in blocks of 16
// rows x 4 inputs (one 64-byte register: lane i holds row i's 4 codes), so one vpdpbusd multiplies 4 inputs of one
// token into 16 rows. Activations are the int8 codes + 128 as unsigned bytes (vpdpbusd is unsigned x signed); the
// 128 x (sum of a row group's codes) this adds is subtracted at the end. A tile is 32 rows x 8 tokens: each weight
// register serves 8 tokens. Integer sums run over one scale group (128 inputs), then are scaled into float.
typedef struct { int rows, rows_pad, K, gsz, ng; signed char *w; float *scale, *off; } TG;

static void tg_from_tm(TG *g, const TMat *m) {  // from a packed TMat: same codes and scales
    g->rows = m->rows; g->rows_pad = (m->rows + 31) / 32 * 32; g->K = m->cols; g->gsz = m->gch * 16; g->ng = m->ng;
    const int K4 = g->K / 4;
    g->w = (signed char *)xalloc((size_t)g->rows_pad * g->K);
    g->scale = (float *)xalloc((size_t)g->rows_pad * g->ng * 4); g->off = (float *)xalloc((size_t)g->rows_pad * 4);
    for (int r = 0; r < m->rows; r++) {
        float off = 0;
        for (int gi = 0; gi < g->ng; gi++) {
            int csum = 0;
            for (int k = gi * g->gsz; k < (gi + 1) * g->gsz; k++) {
                const int bit = k % 16, ch = k / 16;
                const int c = (m->pos[(size_t)r * m->nch + ch] >> bit & 1) - (m->neg[(size_t)r * m->nch + ch] >> bit & 1);
                g->w[(((size_t)(r / 16) * K4 + k / 4) * 16 + r % 16) * 4 + k % 4] = (signed char)c;
                csum += c;
            }
            const float sc = m->scale[(size_t)r * m->ng + gi];
            g->scale[((size_t)(r / 16) * g->ng + gi) * 16 + r % 16] = sc;
            off += 128.0f * csum * sc;
        }
        g->off[r] = off;
    }
}

// out[t * ldo + r] (=, or += with acc) step[t] * sum_k w[r, k] x[t, k], for x the int8 codes + 128 (xu, ldx bytes per token)
static void tg_gemm(const TG *g, const unsigned char *xu, int ldx, const float *step, int T, float *out, int ldo, int acc) {
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
                        const __m512i x = _mm512_set1_epi32(v);
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
                float *o = out + (size_t)(t0 + j) * ldo + r0;
                __m512 v0 = _mm512_mul_ps(_mm512_sub_ps(f0[j], o0), st), v1 = _mm512_mul_ps(_mm512_sub_ps(f1[j], o1), st);
                if (acc) { v0 = _mm512_add_ps(v0, _mm512_maskz_loadu_ps(m0, o)); v1 = _mm512_add_ps(v1, _mm512_maskz_loadu_ps(m1, o + 16)); }
                _mm512_mask_storeu_ps(o, m0, v0); _mm512_mask_storeu_ps(o + 16, m1, v1);
            }
        }
}

static void to_u8(const signed char *q, unsigned char *u, int n) { for (int i = 0; i < n; i++) u[i] = (unsigned char)(q[i] + 128); }

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
static unsigned char *QU, *LQU;
static double t_proj_in, t_scan, t_proj_out, t_trees, t_lin;

static void alloc_scratch(int T) {
    X = (float *)xalloc((size_t)T * D * 4); U = (float *)xalloc((size_t)T * D * 4); OUT = (float *)xalloc((size_t)T * D * 4);
    Z = (float *)xalloc((size_t)T * ZR * 4); DT = (float *)xalloc((size_t)T * H * 4);
    YF = (float *)xalloc((size_t)T * D * 4); YB = (float *)xalloc((size_t)T * D * 4);
    STEP = (float *)xalloc((size_t)T * 4); Q = (signed char *)xalloc((size_t)T * (D + 64));
    QU = (unsigned char *)xalloc((size_t)T * D);
    if (R) { LZ = (float *)xalloc((size_t)T * R * 4); LQ = (signed char *)xalloc((size_t)T * (R + 64)); LQU = (unsigned char *)xalloc((size_t)T * R); }
}

static inline void load_q(const signed char *q, int n, __m512i *xq) { for (int c = 0; c < n / 64; c++) xq[c] = _mm512_loadu_si512(q + 64 * c); }

// --------------------------------------------------------------------------------------------- the BiSSD
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
        float pre[64], tmp[64];
        for (int h = 0; h < H; h++) pre[h] = dot(L->dt_w + (size_t)h * D, u, D) + L->dt_b[h];
        softplus_array(pre, DT + (size_t)t * H, tmp, H);
    }
    if (g_gemm) tg_gemm(&L->gin, QU, D, STEP, T, Z, ZR, 0);
    double t1 = now();
    t_proj_in += t1 - t0;
    // the scans: one (head, direction) per job; the head's state S is N rows of P floats
#pragma omp parallel for schedule(dynamic, 1)
    for (int job = 0; job < 2 * H; job++) {
        const int h = job / 2, dir = job % 2, PV = P / 16;
        __m512 S[64][4];
        for (int n = 0; n < N; n++) for (int v = 0; v < PV; v++) S[n][v] = _mm512_setzero_ps();
        float *Y = dir ? YB : YF;
        for (int k = 0; k < T; k++) {
            const int t = dir ? T - 1 - k : k;
            const float *z = Z + (size_t)t * ZR, *C = z + h * N, *B = z + H * N + h * N, *x = z + 2 * H * N + h * P;
            const float dt = DT[(size_t)t * H + h];
            const __m512 a = _mm512_set1_ps(expf(dt * L->A[h]));
            __m512 xv[4], y[4];
            for (int v = 0; v < PV; v++) { xv[v] = _mm512_loadu_ps(x + 16 * v); y[v] = _mm512_setzero_ps(); }
            if (!dir) {
                for (int n = 0; n < N; n++) {
                    const __m512 b = _mm512_set1_ps(dt * B[n]), c = _mm512_set1_ps(C[n]);
                    for (int v = 0; v < PV; v++) { S[n][v] = _mm512_fmadd_ps(S[n][v], a, _mm512_mul_ps(b, xv[v])); y[v] = _mm512_fmadd_ps(c, S[n][v], y[v]); }
                }
            } else {
                for (int n = 0; n < N; n++) {
                    const __m512 b = _mm512_set1_ps(dt * B[n]), c = _mm512_set1_ps(C[n]);
                    for (int v = 0; v < PV; v++) { S[n][v] = _mm512_mul_ps(S[n][v], a); y[v] = _mm512_fmadd_ps(c, S[n][v], y[v]); S[n][v] = _mm512_fmadd_ps(b, xv[v], S[n][v]); }
                }
            }
            for (int v = 0; v < PV; v++) _mm512_storeu_ps(Y + (size_t)t * D + h * P + 16 * v, y[v]);
        }
    }
    double t2 = now();
    t_scan += t2 - t1;
#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {  // y = forward + backward + D x; int8; out_proj; residual
        float y[1024];
        const float *x = Z + (size_t)t * ZR + 2 * H * N;
        for (int c = 0; c < D; c++) y[c] = YF[(size_t)t * D + c] + YB[(size_t)t * D + c] + L->Dsk[c / P] * x[c];
        signed char *q = Q + (size_t)t * (D + 64);
        const float st = STEP[t] = quant8(y, q, D);
        if (g_gemm) { to_u8(q, QU + (size_t)t * D, D); continue; }
        __m512i xq[16];
        load_q(q, D, xq);
        float *h = X + (size_t)t * D;
        for (int r = 0; r < D; r++) h[r] += st * tm_dot_i8(&L->out, r, xq);
    }
    if (g_gemm) tg_gemm(&L->gout, QU, D, STEP, T, X, D, 1);
    t_proj_out += now() - t2;
}

// ------------------------------------------------------------------------- deep trees and the linear path
static void ffn(const Layer *L, int T) {
    double t0 = now(), tl = 0;
#pragma omp parallel for schedule(static) reduction(+ : tl)
    for (int t = 0; t < T; t++) {
        float *u = U + (size_t)t * D, *o = OUT + (size_t)t * D;
        layernorm(X + (size_t)t * D, L->norm2, u, D);
        signed char *q = Q + (size_t)t * (D + 64);
        const float st = STEP[t] = quant8(u, q, D);
        __m512i xq[16];
        load_q(q, D, xq);
        int rows[256], S = 0;
        float lg[256], coef[256];
        for (int tr = 0; tr < NT; tr++) {
            int cur = 0;
            for (int k = 0; k <= DEPTH; k++) {
                const int gid = tr * PER + cur;
                const float l = st * tm_dot_i8(&L->node_in, gid, xq) + L->node_bias[gid];
                rows[S] = gid; lg[S++] = l;
                cur = 2 * cur + 1 + (l > 0);
            }
        }
        gelu_array(lg, coef, S);
        memset(o, 0, D * 4);
        tm_out_rows(&L->node_out, rows, coef, S, o);
        if (R && g_gemm) to_u8(q, QU + (size_t)t * D, D);
        if (R && !g_gemm) {
            const double s0 = now();
            float *lz = LZ + (size_t)t * R;
            for (int r = 0; r < R; r++) lz[r] = st * tm_dot_i8(&L->ldown, r, xq);
            signed char *lq = LQ + (size_t)t * (R + 64);
            const float s2 = quant8(lz, lq, R);
            __m512i zq[16];
            load_q(lq, R, zq);
            for (int c = 0; c < D; c++) o[c] += s2 * tm_dot_i8(&L->lup, c, zq) + L->lbias[c];
            tl += now() - s0;
        }
        if (!(R && g_gemm)) { float *h = X + (size_t)t * D; for (int c = 0; c < D; c++) h[c] += o[c]; }
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
        tl += (now() - s0) * (
#ifdef _OPENMP
            omp_get_max_threads()
#else
            1
#endif
        );
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
static void synth_layer(Layer *L, int idx) {
    L->has_norm1 = idx > 0;
    L->norm1 = frnd(D, 0); for (int c = 0; c < D; c++) L->norm1[c] = 1;
    L->norm2 = L->norm1;
    tm_random(&L->in, ZR, D); tm_random(&L->out, D, D);
    L->dt_w = frnd((size_t)H * D, 0.01f); L->dt_b = frnd(H, 0.1f); L->A = frnd(H, 0.0f); L->Dsk = frnd(H, 1.0f);
    for (int h = 0; h < H; h++) L->A[h] = -0.2f;
    tm_random(&L->node_in, NN, D); tm_random(&L->node_out, NN, D); L->node_bias = frnd(NN, 0.1f);
    if (R) { tm_random(&L->ldown, R, D); tm_random(&L->lup, D, R); L->lbias = frnd(D, 0.1f); }
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
        t_proj_in = t_scan = t_proj_out = t_trees = t_lin = 0;
        for (int r = 0; r < reps; r++) { const double t0 = now(); encode(seq, L, hidden); const double dt = now() - t0; if (dt < best) best = dt; }
        const double tot = t_proj_in + t_scan + t_proj_out + t_trees + t_lin;
        printf("  sequence %3d: %.2f ms (best of %d) = %.1f us per token = %.0f tokens/s.  Share: in_proj %.0f%%, scans %.0f%%, "
               "out_proj %.0f%%, trees %.0f%%, linear path %.0f%%\n", L, best * 1e3, reps, best / L * 1e6, L / best,
               100 * t_proj_in / tot, 100 * t_scan / tot, 100 * t_proj_out / tot, 100 * t_trees / tot, 100 * t_lin / tot);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: gtsl_bench gtsl.bin [repeats] | gtsl_bench synth [layers] [linear_rank]\n"); return 1; }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    if (getenv("GTSL_ROWDOT")) g_gemm = 0;
    if (!strcmp(argv[1], "synth")) {
        V = 50368; D = 1024; NL = argc > 2 ? atoi(argv[2]) : 28; H = 16; N = 64; P = 64; NT = 4; DEPTH = 9;
        R = argc > 3 ? atoi(argv[3]) : 0; EPS = 1e-5f;
        set_shape();
        emb = frnd((size_t)V * D, 0.05f); emb_norm = frnd(D, 0); norm_f = emb_norm;
        for (int c = 0; c < D; c++) emb_norm[c] = 1;
        layers = (Layer *)xalloc(NL * sizeof(Layer));
        for (int l = 0; l < NL; l++) { synth_layer(&layers[l], l); build_gemm(&layers[l]); }
        alloc_scratch(512);
        printf("%d thread(s). Synthetic GTS-L: %d layers, width %d, BiSSD %d heads x (state %d, head %d), %d trees of depth %d, linear rank %d\n",
               threads, NL, D, H, N, P, NT, DEPTH, R);
        int ids[512];
        for (int t = 0; t < 512; t++) ids[t] = 1000 + t;
        float *hidden = (float *)xalloc((size_t)512 * D * 4);
        time_it(ids, 512, 3, hidden);
        return 0;
    }
    const int reps = argc > 2 ? atoi(argv[2]) : 3;
    g_f = fopen(argv[1], "rb"); if (!g_f) { perror("open"); return 1; }
    if (rdi() != 11) { fprintf(stderr, "not a GTS-L file\n"); return 1; }
    V = rdi(); D = rdi(); NL = rdi(); H = rdi(); N = rdi(); P = rdi(); NT = rdi(); DEPTH = rdi(); R = rdi();
    if (fread(&EPS, 4, 1, g_f) != 1) return 1;
    set_shape();
    if (D % 128 || D > 1024 || N > 64 || P != 64 || H > 64 || H * P != D || (R && (R % 64 || R > 1024))) { fprintf(stderr, "unsupported shape\n"); return 1; }
    emb = rdf((size_t)V * D); emb_norm = rdf(D); norm_f = rdf(D); head_dense = rdf((size_t)D * D); head_norm = rdf(D); dec_bias = rdf(V);
    layers = (Layer *)xalloc(NL * sizeof(Layer));
    for (int l = 0; l < NL; l++) { load_layer(&layers[l]); build_gemm(&layers[l]); }
    const int T = rdi(), M = rdi();
    int *ids = (int *)xalloc(T * 4), *pos = (int *)xalloc(M * 4);
    if (fread(ids, 4, T, g_f) != (size_t)T || fread(pos, 4, M, g_f) != (size_t)M) return 1;
    float *ref = rdf((size_t)M * V);
    fclose(g_f);
    alloc_scratch(512 > T ? 512 : T);
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
