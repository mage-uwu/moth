// Golden Tree Snake (GTS) fork, 2026.
// Token-at-a-time autoregressive inference for the two models trained by scripts/shakespeare_ar.py:
// a causal GTS language model and a Mamba-2 language model. Loads model.bin, checks its logits against
// the PyTorch logits stored in the file, then times it.
//
//   gcc -O3 -march=native -ffast-math -funroll-loops ar_bench.c -o ar_bench -lm
//   ./ar_bench runs/gts/model.bin [n_tokens]
//
// The file holds ternary weights as scale * {-1, 0, +1} floats. With AVX-512 they are packed at load time (below);
// the third argument "float" runs them as plain float32 instead, and "m2rows" runs Mamba-2's float projections as
// row dot products rather than in column form (the kernel before the column form was added).
#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static void *xalloc(size_t bytes) { void *p; if (posix_memalign(&p, 64, bytes ? bytes : 64)) { fprintf(stderr, "oom\n"); exit(1); } memset(p, 0, bytes); return p; }
static FILE *g_f;
static float *rdf(size_t n) { float *p = (float *)xalloc(n * sizeof(float)); if (fread(p, sizeof(float), n, g_f) != n) { fprintf(stderr, "short read\n"); exit(1); } return p; }
static int rdi(void) { int v; if (fread(&v, sizeof(int), 1, g_f) != 1) { fprintf(stderr, "short read\n"); exit(1); } return v; }

static inline float dot(const float *a, const float *b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static inline void axpy(float alpha, const float *x, float *y, int n) { for (int i = 0; i < n; i++) y[i] += alpha * x[i]; }
static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
static inline float geluf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
static inline float siluf(float x) { return x / (1.0f + expf(-x)); }
// The per-element maths below is written as plain loops over arrays so the compiler can vectorise the exp calls.
// erf by Abramowitz & Stegun 7.1.26 (absolute error under 1.5e-7), because libm's erff is scalar only.
static inline float erf_as(float x) {
    const float ax = fabsf(x), t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * expf(-ax * ax);
    return x < 0 ? -y : y;
}
static void gelu_array(const float *x, float *y, int n) { for (int i = 0; i < n; i++) y[i] = 0.5f * x[i] * (1.0f + erf_as(x[i] * 0.70710678f)); }
static void exp_array(const float *x, float *y, int n) { for (int i = 0; i < n; i++) y[i] = expf(x[i]); }
static void silu_array(float *x, float *tmp, int n) { for (int i = 0; i < n; i++) tmp[i] = expf(-x[i]); for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + tmp[i]); }
static void softplus_array(const float *x, float *y, float *tmp, int n) { exp_array(x, tmp, n); for (int i = 0; i < n; i++) y[i] = x[i] > 20.0f ? x[i] : logf(1.0f + tmp[i]); }
static void rmsnorm_eps(const float *x, const float *w, float *y, int n, float eps) {
    float s = 0; for (int i = 0; i < n; i++) s += x[i] * x[i];
    s = 1.0f / sqrtf(s / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * s * w[i];
}

static void rmsnorm(const float *x, const float *w, float *y, int n) { rmsnorm_eps(x, w, y, n, 1e-5f); }

// ------------------------------------------------------------------------ packed ternary rows
// A ternary weight row is stored as two bit masks per 16 weights (which are +1, which are -1) and one scale per
// group of weights: 32 bytes for 128 weights instead of 512. A dot product is then masked adds over the input held
// in registers, and a scaled add of the row is masked adds into the output. Same function as the float path, up
// to summation order. The file stores effective float weights (scale * code); packing recovers the codes at load
// time and refuses any tensor that is not exactly ternary per group.
static int g_use_pack = 1, g_packed = 0, g_float = 0, g_i16 = 0, g_colform = 1;
typedef struct { int rows, cols, nch, gch, ng; unsigned short *pos, *neg; float *scale; } TMat;
#ifdef __AVX512F__
static int tm_pack(TMat *m, const float *w, int rows, int cols) {
    int gsz = cols < 128 ? cols : 128;
    while (cols % gsz) gsz--; // the training code's group size: the largest divisor of the row length up to 128
    if (!g_use_pack || cols % 16 || gsz % 32) { g_float++; return 0; }
    m->rows = rows; m->cols = cols; m->nch = cols / 16; m->gch = gsz / 16; m->ng = cols / gsz;
    m->pos = (unsigned short *)xalloc((size_t)rows * m->nch * 2); m->neg = (unsigned short *)xalloc((size_t)rows * m->nch * 2);
    m->scale = (float *)xalloc((size_t)rows * m->ng * sizeof(float));
    for (int r = 0; r < rows; r++)
        for (int g = 0; g < m->ng; g++) {
            const float *v = w + (size_t)r * cols + g * gsz;
            float sc = 0;
            for (int j = 0; j < gsz; j++) if (fabsf(v[j]) > sc) sc = fabsf(v[j]);
            for (int j = 0; j < gsz; j++) {
                if (v[j] != 0 && fabsf(fabsf(v[j]) - sc) > 1e-5f * sc) { g_float++; return 0; } // not ternary
                if (v[j] > 0) m->pos[(size_t)r * m->nch + (g * gsz + j) / 16] |= 1u << (j % 16);
                if (v[j] < 0) m->neg[(size_t)r * m->nch + (g * gsz + j) / 16] |= 1u << (j % 16);
            }
            m->scale[(size_t)r * m->ng + g] = sc;
        }
    g_packed++;
    return 1;
}
static inline void tm_load(const float *x, __m512 *xv, int nch) { for (int c = 0; c < nch; c++) xv[c] = _mm512_loadu_ps(x + 16 * c); }
static inline float tm_dot(const TMat *m, int r, const __m512 *xv) {
    const unsigned short *p = m->pos + (size_t)r * m->nch, *n = m->neg + (size_t)r * m->nch;
    __m512 tv = _mm512_setzero_ps();
    for (int g = 0; g < m->ng; g++) {
        __m512 p0 = _mm512_setzero_ps(), p1 = p0, n0 = p0, n1 = p0; // four short chains instead of one long one
        for (int c = g * m->gch; c < (g + 1) * m->gch; c += 2) {
            p0 = _mm512_mask_add_ps(p0, p[c], p0, xv[c]); n0 = _mm512_mask_add_ps(n0, n[c], n0, xv[c]);
            p1 = _mm512_mask_add_ps(p1, p[c + 1], p1, xv[c + 1]); n1 = _mm512_mask_add_ps(n1, n[c + 1], n1, xv[c + 1]);
        }
        tv = _mm512_fmadd_ps(_mm512_set1_ps(m->scale[(size_t)r * m->ng + g]), _mm512_sub_ps(_mm512_add_ps(p0, p1), _mm512_add_ps(n0, n1)), tv);
    }
    return _mm512_reduce_add_ps(tv); // one horizontal sum per row, not one per group
}
static inline void tm_axpy(const TMat *m, int r, float alpha, __m512 *yv) {
    const unsigned short *p = m->pos + (size_t)r * m->nch, *n = m->neg + (size_t)r * m->nch;
    for (int g = 0; g < m->ng; g++) {
        const __m512 a = _mm512_set1_ps(alpha * m->scale[(size_t)r * m->ng + g]);
        for (int c = g * m->gch; c < (g + 1) * m->gch; c++) { yv[c] = _mm512_mask_add_ps(yv[c], p[c], yv[c], a); yv[c] = _mm512_mask_sub_ps(yv[c], n[c], yv[c], a); }
    }
}
// out += sum_i coef[i] * row(rows[i]). For 128-wide rows with one scale the eight output registers stay in
// registers for the whole sum, in two sets fed by alternating rows so that consecutive rows do not wait on each other.
#define TM_AX(Y, c) { Y = _mm512_mask_add_ps(Y, p[c], Y, a); Y = _mm512_mask_sub_ps(Y, n[c], Y, a); }
static void tm_out_rows(const TMat *m, const int *rows, const float *coef, int S, float *out) {
    if (m->gch == 8) { // one scale per 128 weights: do the output 128 columns at a time, each block entirely in registers
        for (int b = 0; b < m->ng; b++) {
            float *o = out + 128 * b;
            __m512 y0 = _mm512_loadu_ps(o), y1 = _mm512_loadu_ps(o + 16), y2 = _mm512_loadu_ps(o + 32), y3 = _mm512_loadu_ps(o + 48);
            __m512 y4 = _mm512_loadu_ps(o + 64), y5 = _mm512_loadu_ps(o + 80), y6 = _mm512_loadu_ps(o + 96), y7 = _mm512_loadu_ps(o + 112);
            __m512 z0 = _mm512_setzero_ps(), z1 = z0, z2 = z0, z3 = z0, z4 = z0, z5 = z0, z6 = z0, z7 = z0;
            int i = 0;
            for (; i + 2 <= S; i += 2) {
                { const unsigned short *p = m->pos + (size_t)rows[i] * m->nch + 8 * b, *n = m->neg + (size_t)rows[i] * m->nch + 8 * b;
                  const __m512 a = _mm512_set1_ps(coef[i] * m->scale[(size_t)rows[i] * m->ng + b]);
                  TM_AX(y0, 0) TM_AX(y1, 1) TM_AX(y2, 2) TM_AX(y3, 3) TM_AX(y4, 4) TM_AX(y5, 5) TM_AX(y6, 6) TM_AX(y7, 7) }
                { const unsigned short *p = m->pos + (size_t)rows[i + 1] * m->nch + 8 * b, *n = m->neg + (size_t)rows[i + 1] * m->nch + 8 * b;
                  const __m512 a = _mm512_set1_ps(coef[i + 1] * m->scale[(size_t)rows[i + 1] * m->ng + b]);
                  TM_AX(z0, 0) TM_AX(z1, 1) TM_AX(z2, 2) TM_AX(z3, 3) TM_AX(z4, 4) TM_AX(z5, 5) TM_AX(z6, 6) TM_AX(z7, 7) }
            }
            if (i < S) { const unsigned short *p = m->pos + (size_t)rows[i] * m->nch + 8 * b, *n = m->neg + (size_t)rows[i] * m->nch + 8 * b;
                  const __m512 a = _mm512_set1_ps(coef[i] * m->scale[(size_t)rows[i] * m->ng + b]);
                  TM_AX(y0, 0) TM_AX(y1, 1) TM_AX(y2, 2) TM_AX(y3, 3) TM_AX(y4, 4) TM_AX(y5, 5) TM_AX(y6, 6) TM_AX(y7, 7) }
            _mm512_storeu_ps(o, _mm512_add_ps(y0, z0)); _mm512_storeu_ps(o + 16, _mm512_add_ps(y1, z1)); _mm512_storeu_ps(o + 32, _mm512_add_ps(y2, z2)); _mm512_storeu_ps(o + 48, _mm512_add_ps(y3, z3));
            _mm512_storeu_ps(o + 64, _mm512_add_ps(y4, z4)); _mm512_storeu_ps(o + 80, _mm512_add_ps(y5, z5)); _mm512_storeu_ps(o + 96, _mm512_add_ps(y6, z6)); _mm512_storeu_ps(o + 112, _mm512_add_ps(y7, z7));
        }
        return;
    }
    __m512 yv[64];
    tm_load(out, yv, m->nch);
    for (int i = 0; i < S; i++) tm_axpy(m, rows[i], coef[i], yv);
    memcpy(out, yv, (size_t)m->cols * sizeof(float));
}
// Column form of a packed ternary matrix for dense matrix-vector products y = W x (every row needed, as in Mamba-2's
// projections). Rows go in blocks of up to 256 (16 registers of 16 rows). Within a block, for each input column c,
// the masks of its rows are stored together: pos[block][c][j], j = register. One broadcast of x[c] is added into all
// sixteen registers under those masks, so there is no horizontal sum, and sixteen independent chains.
// Per-row scales (one per 128-column group) are applied once per group: t[j] holds the group's partial sums.
typedef struct { int rows, cols, gsz, ng, nblk; unsigned short *pos, *neg; float *scale_t; } CMat; // scale_t[g * rows + r]
static int cm_from_tm(CMat *c, const TMat *m) {
    if (m->rows % 16) return 0;
    c->rows = m->rows; c->cols = m->cols; c->gsz = m->gch * 16; c->ng = m->ng; c->nblk = (m->rows + 255) / 256;
    c->pos = (unsigned short *)xalloc((size_t)m->rows / 16 * m->cols * 2); c->neg = (unsigned short *)xalloc((size_t)m->rows / 16 * m->cols * 2);
    c->scale_t = (float *)xalloc((size_t)m->ng * m->rows * sizeof(float));
    for (int r = 0; r < m->rows; r++) for (int g = 0; g < m->ng; g++) c->scale_t[(size_t)g * m->rows + r] = m->scale[(size_t)r * m->ng + g];
    for (int b = 0; b < c->nblk; b++) {
        const int r0 = 256 * b, nj = (m->rows - r0) / 16 < 16 ? (m->rows - r0) / 16 : 16;
        unsigned short *P = c->pos + (size_t)r0 / 16 * m->cols, *N = c->neg + (size_t)r0 / 16 * m->cols;
        for (int col = 0; col < m->cols; col++)
            for (int j = 0; j < nj; j++) {
                unsigned short pw = 0, nw = 0;
                for (int l = 0; l < 16; l++) {
                    const int r = r0 + 16 * j + l;
                    pw |= ((m->pos[(size_t)r * m->nch + col / 16] >> (col % 16)) & 1u) << l;
                    nw |= ((m->neg[(size_t)r * m->nch + col / 16] >> (col % 16)) & 1u) << l;
                }
                P[(size_t)col * nj + j] = pw; N[(size_t)col * nj + j] = nw;
            }
    }
    return 1;
}
#define CM_STEP(J) { t##J = _mm512_mask_add_ps(t##J, P[J], t##J, xb); t##J = _mm512_mask_sub_ps(t##J, N[J], t##J, xb); }
static void cm_matvec(const CMat *c, const float *x, float *y) {
    for (int b = 0; b < c->nblk; b++) {
        const int r0 = 256 * b, nj = (c->rows - r0) / 16 < 16 ? (c->rows - r0) / 16 : 16;
        __m512 acc[16];
        for (int j = 0; j < nj; j++) acc[j] = _mm512_setzero_ps();
        for (int g = 0; g < c->ng; g++) {
            __m512 t0 = _mm512_setzero_ps(), t1 = t0, t2 = t0, t3 = t0, t4 = t0, t5 = t0, t6 = t0, t7 = t0, t8 = t0, t9 = t0, t10 = t0, t11 = t0, t12 = t0, t13 = t0, t14 = t0, t15 = t0;
            const unsigned short *P = c->pos + (size_t)r0 / 16 * c->cols + (size_t)g * c->gsz * nj, *N = c->neg + (size_t)r0 / 16 * c->cols + (size_t)g * c->gsz * nj;
            if (nj == 16) {
                for (int col = g * c->gsz; col < (g + 1) * c->gsz; col++, P += 16, N += 16) {
                    const __m512 xb = _mm512_set1_ps(x[col]);
                    CM_STEP(0) CM_STEP(1) CM_STEP(2) CM_STEP(3) CM_STEP(4) CM_STEP(5) CM_STEP(6) CM_STEP(7)
                    CM_STEP(8) CM_STEP(9) CM_STEP(10) CM_STEP(11) CM_STEP(12) CM_STEP(13) CM_STEP(14) CM_STEP(15)
                }
            } else {
                __m512 t[16];
                for (int j = 0; j < nj; j++) t[j] = _mm512_setzero_ps();
                for (int col = g * c->gsz; col < (g + 1) * c->gsz; col++, P += nj, N += nj) {
                    const __m512 xb = _mm512_set1_ps(x[col]);
                    for (int j = 0; j < nj; j++) { t[j] = _mm512_mask_add_ps(t[j], P[j], t[j], xb); t[j] = _mm512_mask_sub_ps(t[j], N[j], t[j], xb); }
                }
                t0 = t[0]; t1 = t[1]; t2 = t[2]; t3 = t[3]; t4 = t[4]; t5 = t[5]; t6 = t[6]; t7 = t[7];
                t8 = t[8]; t9 = t[9]; t10 = t[10]; t11 = t[11]; t12 = t[12]; t13 = t[13]; t14 = t[14]; t15 = t[15];
            }
            const __m512 tv[16] = {t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15};
            const float *sc = c->scale_t + (size_t)g * c->rows + r0;
            for (int j = 0; j < nj; j++) acc[j] = _mm512_fmadd_ps(_mm512_loadu_ps(sc + 16 * j), tv[j], acc[j]);
        }
        for (int j = 0; j < nj; j++) _mm512_storeu_ps(y + r0 + 16 * j, acc[j]);
    }
}
#endif
// 8-bit activations: when the input of a ternary layer was trained quantised to int8 per token, a row's dot product
// is an exact integer sum of +x, -x or 0 over bytes, 64 at a time.
static void quant_float(const float *x, float *y, int n, int bits) { // BitNet's activation_quant, in float
    const float qmax = (float)((1 << (bits - 1)) - 1);
    float mx = 0; for (int c = 0; c < n; c++) { const float a = fabsf(x[c]); if (a > mx) mx = a; }
    const float sc = qmax / (mx > 1e-5f ? mx : 1e-5f), inv = 1.0f / sc;
    for (int c = 0; c < n; c++) { float q = rintf(x[c] * sc); q = q > qmax ? qmax : q < -qmax - 1 ? -qmax - 1 : q; y[c] = q * inv; }
}
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define HAVE_I8 1
static float quant8(const float *x, signed char *q, int n) { // int8 codes in [-127, 127]; returns the step size
    float mx = 0; for (int c = 0; c < n; c++) { const float a = fabsf(x[c]); if (a > mx) mx = a; }
    const float sc = 127.0f / (mx > 1e-5f ? mx : 1e-5f);
    int c = 0;
    const __m512 scv = _mm512_set1_ps(sc);
    for (; c + 16 <= n; c += 16) _mm_storeu_si128((__m128i *)(q + c), _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(x + c), scv))));
    for (; c < n; c++) q[c] = (signed char)rintf(x[c] * sc);
    return 1.0f / sc;
}
static inline float tm_dot_i8(const TMat *m, int r, const __m512i *xq) {
    const unsigned short *p = m->pos + (size_t)r * m->nch, *n = m->neg + (size_t)r * m->nch;
    const __m512i ones8 = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1), zero = _mm512_setzero_si512();
    float total = 0;
    for (int g = 0; g < m->ng; g++) {
        __m512i acc = zero;
        for (int c = g * m->gch; c < (g + 1) * m->gch; c += 4) { // 4 mask words = 64 weights
            unsigned long long kp, kn; memcpy(&kp, p + c, 8); memcpy(&kn, n + c, 8);
            const __m512i xb = xq[c / 4];
            const __m512i t = _mm512_mask_sub_epi8(_mm512_maskz_mov_epi8(kp, xb), kn, zero, xb); // +x, -x or 0
#ifdef __AVX512VNNI__
            acc = _mm512_dpbusd_epi32(acc, ones8, t); (void)ones16;
#else
            acc = _mm512_add_epi32(acc, _mm512_madd_epi16(_mm512_maddubs_epi16(ones8, t), ones16));
#endif
        }
        total += m->scale[(size_t)r * m->ng + g] * (float)_mm512_reduce_add_epi32(acc);
    }
    return total;
}
// Output of a mixer whose input was int8: sum the path's rows with 16-bit integer coefficients, 32 weights per
// masked add instead of 16. The coefficients share one step, sized so that no partial sum can overflow.
static void tm_out_i16(const TMat *m, const int *rows, const float *coef, int S, float *out) {
    float tot = 0;
    for (int i = 0; i < S; i++) {
        float mx = 0;
        for (int g = 0; g < m->ng; g++) { const float a = fabsf(coef[i] * m->scale[(size_t)rows[i] * m->ng + g]); if (a > mx) mx = a; }
        tot += mx;
    }
    const float q = tot > 0 ? tot / 32000.0f : 1.0f, iq = 1.0f / q;
    const int n32 = m->cols / 32;
    __m512i acc[32];
    for (int c = 0; c < n32; c++) acc[c] = _mm512_setzero_si512();
    for (int i = 0; i < S; i++) {
        const unsigned short *p = m->pos + (size_t)rows[i] * m->nch, *n = m->neg + (size_t)rows[i] * m->nch;
        for (int g = 0; g < m->ng; g++) {
            const __m512i kv = _mm512_set1_epi16((short)rintf(coef[i] * m->scale[(size_t)rows[i] * m->ng + g] * iq));
            for (int c = g * m->gch; c < (g + 1) * m->gch; c += 2) {
                unsigned int kp, kn; memcpy(&kp, p + c, 4); memcpy(&kn, n + c, 4);
                const __m512i a = _mm512_mask_add_epi16(acc[c / 2], kp, acc[c / 2], kv);
                acc[c / 2] = _mm512_mask_sub_epi16(a, kn, a, kv);
            }
        }
    }
    const __m512 qv = _mm512_set1_ps(q);
    for (int c = 0; c < n32; c++) {
        _mm512_storeu_ps(out + 32 * c, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_castsi512_si256(acc[c]))), qv));
        _mm512_storeu_ps(out + 32 * c + 16, _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(acc[c], 1))), qv));
    }
}
#else
#define HAVE_I8 0
#endif
#ifndef __AVX512F__
typedef struct { float v[16]; } __m512_stub;
#define __m512 __m512_stub
static int tm_pack(TMat *m, const float *w, int rows, int cols) { (void)m; (void)w; (void)rows; (void)cols; g_float++; return 0; }
static inline void tm_load(const float *x, __m512 *xv, int nch) { (void)x; (void)xv; (void)nch; }
static inline float tm_dot(const TMat *m, int r, const __m512 *xv) { (void)m; (void)r; (void)xv; return 0; }
static inline void tm_axpy(const TMat *m, int r, float alpha, __m512 *yv) { (void)m; (void)r; (void)alpha; (void)yv; }
static void tm_out_rows(const TMat *m, const int *rows, const float *coef, int S, float *out) { (void)m; (void)rows; (void)coef; (void)S; (void)out; }
typedef struct { int rows; } CMat;
static int cm_from_tm(CMat *c, const TMat *m) { (void)c; (void)m; return 0; }
static void cm_matvec(const CMat *c, const float *x, float *y) { (void)c; (void)x; (void)y; }
#endif

// ------------------------------------------------------------------------------ causal GTS

typedef struct {
    int d, K, N, G, n_nodes, kc, read_state, write_logit, n_trees, n_heads, act; // act: 0 gelu(l + c), 1 l + c, 2 gelu(l) + c
    float *node_in, *node_bias, *node_out, *conv_w, *conv_b, *ctx_w, *dt_bias, *A, *read_norm_w, *read_w;
    float *h, *stamp, *hist, *x, *p, *found, *sl, *sd, *sf, *sc, *se; int *sg, *sq; unsigned char *visited; float clock[256]; TMat t_in, t_out, t_ctx; int packed, act_bits; float *conv_t; signed char *xq;
} GLayer;

static GLayer *gts_load(int d, int depth, int N, int kc, int read_state, int write_logit, int n_trees, int act, int n_heads, int act_bits, int use_ctx) {
    GLayer *g = (GLayer *)xalloc(sizeof(GLayer));
    if (!use_ctx) N = 0; // a stateless mixer (plain FFF trees): no keys, queries, clocks or node states
    g->d = d; g->K = depth + 1; g->N = N; g->kc = kc; g->n_trees = n_trees; g->act = act; g->n_heads = n_heads; g->G = use_ctx ? n_heads * (depth + 1) : 0; // one clock per (head, level)
    g->n_nodes = n_trees * ((1 << (depth + 1)) - 1); // all trees, stored one after another
    g->node_in = rdf((size_t)g->n_nodes * d); g->node_bias = rdf(g->n_nodes); g->node_out = rdf((size_t)g->n_nodes * d);
    g->conv_w = rdf(d * kc); g->conv_b = rdf(d); g->act_bits = act_bits;
    g->xq = (signed char *)xalloc(d + 64);
    g->conv_t = (float *)xalloc((size_t)kc * d * sizeof(float)); // tap-major copy: conv_t[j * d + c] = weight of tap j, channel c
    for (int c = 0; c < d; c++) for (int j = 0; j < kc; j++) g->conv_t[(size_t)j * d + c] = g->conv_w[c * kc + j];
    g->ctx_w = rdf((size_t)(2 * N + g->G) * d); // rows: B, C, dt
    g->dt_bias = rdf(g->G); g->A = rdf(g->G);
    for (int k = 0; k < g->G; k++) g->A[k] = -expf(g->A[k]); // file holds A_log
    g->read_state = read_state; g->write_logit = write_logit;
    if (read_state) { g->read_norm_w = rdf(n_trees * g->K * N); g->read_w = rdf((size_t)d * n_trees * g->K * N); }
    g->found = (float *)xalloc(n_trees * g->K * N * sizeof(float));
    { // per-slot scratch; slot = tree * K + level, and sq[slot] is the slot's clock (its head and level)
        const int S = n_trees * g->K, sz = (S > g->G ? S : g->G) * sizeof(float);
        g->sl = (float *)xalloc(sz); g->sd = (float *)xalloc(sz); g->sf = (float *)xalloc(sz); g->sc = (float *)xalloc(sz); g->se = (float *)xalloc(sz);
        g->sg = (int *)xalloc(S * sizeof(int)); g->sq = (int *)xalloc(S * sizeof(int));
        for (int t = 0; t < n_trees; t++) for (int k = 0; k < g->K; k++) g->sq[t * g->K + k] = (t * n_heads / n_trees) * g->K + k;
    }
    g->h = (float *)xalloc((size_t)g->n_nodes * N * sizeof(float)); g->stamp = (float *)xalloc(g->n_nodes * sizeof(float));
    g->visited = (unsigned char *)xalloc(g->n_nodes);
    g->hist = (float *)xalloc((size_t)(kc - 1) * d * sizeof(float)); g->x = (float *)xalloc(d * sizeof(float)); g->p = (float *)xalloc((2 * N + g->G) * sizeof(float));
    g->packed = tm_pack(&g->t_in, g->node_in, g->n_nodes, d) && tm_pack(&g->t_out, g->node_out, g->n_nodes, d) && tm_pack(&g->t_ctx, g->ctx_w, 2 * N + g->G, d);
    return g;
}

static void gts_reset(GLayer *g) {
    memset(g->visited, 0, g->n_nodes); memset(g->hist, 0, (size_t)(g->kc - 1) * g->d * sizeof(float)); memset(g->clock, 0, sizeof(g->clock));
}

// One token. Walk the tree once; at each node on the path read its lazily decayed state, then write to it.
static void gts_step(GLayer *g, const float *u, float *out) {
    const int d = g->d, K = g->K, N = g->N, kc = g->kc;
    { // causal depthwise conv over the last kc inputs
        const float *wl = g->conv_t + (size_t)(kc - 1) * d;
        for (int c = 0; c < d; c++) g->x[c] = wl[c] * u[c] + g->conv_b[c];
        for (int j = 0; j < kc - 1; j++) { const float *wj = g->conv_t + (size_t)j * d, *hj = g->hist + (size_t)j * d; for (int c = 0; c < d; c++) g->x[c] += wj[c] * hj[c]; }
    }
    int i8 = 0; float step = 0; // per-token absmax quantisation of the mixer input, as trained
#if HAVE_I8
    __m512i xq[16];
    if (g->act_bits == 8 && g->packed && d % 64 == 0 && g->t_in.gch % 4 == 0 && d <= 1024) {
        i8 = 1; step = quant8(g->x, g->xq, d);
        for (int c = 0; c < d / 64; c++) xq[c] = _mm512_loadu_si512(g->xq + 64 * c);
    }
#endif
    if (g->act_bits && !i8) quant_float(g->x, g->x, d, g->act_bits);
    if (kc > 1) { memmove(g->hist, g->hist + d, (size_t)(kc - 2) * d * sizeof(float)); memcpy(g->hist + (size_t)(kc - 2) * d, u, d * sizeof(float)); }
    const float *x = g->x;
    __m512 xv[256];
#if HAVE_I8
    if (i8) { for (int j = 0; j < 2 * N + g->G; j++) g->p[j] = step * tm_dot_i8(&g->t_ctx, j, xq); } else
#endif
    if (g->packed) { tm_load(x, xv, d / 16); for (int j = 0; j < 2 * N + g->G; j++) g->p[j] = tm_dot(&g->t_ctx, j, xv); }
    else for (int j = 0; j < 2 * N + g->G; j++) g->p[j] = dot(g->ctx_w + (size_t)j * d, x, d);
    const float *B = g->p, *C = g->p + N;
    const int T = g->n_trees, S = T * K, per_tree = g->n_nodes / T;
    float dts[256], tmp[256];
    for (int q = 0; q < g->G; q++) tmp[q] = g->p[2 * N + q] + g->dt_bias[q];
    softplus_array(tmp, dts, g->se, g->G);
    for (int q = 0; q < g->G; q++) g->clock[q] += dts[q] * g->A[q];

    // 1. walk: level by level, all trees at once (a level only depends on the level above)
    int cur[256];
    for (int t = 0; t < T; t++) cur[t] = 0;
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) {
            const int gid = t * per_tree + cur[t];
#if HAVE_I8
            const float lg = (i8 ? step * tm_dot_i8(&g->t_in, gid, xq) : g->packed ? tm_dot(&g->t_in, gid, xv) : dot(x, g->node_in + (size_t)gid * d, d)) + g->node_bias[gid];
#else
            const float lg = (g->packed ? tm_dot(&g->t_in, gid, xv) : dot(x, g->node_in + (size_t)gid * d, d)) + g->node_bias[gid];
#endif
            g->sg[t * K + k] = gid; g->sl[t * K + k] = lg;
            cur[t] = 2 * cur[t] + 1 + (lg > 0);
            if (g->packed) { // large models: start fetching this node's read row and the next node's write row
                const size_t ro = (size_t)gid * g->t_out.nch, bytes = (size_t)g->t_out.nch * 2;
                for (size_t o = 0; o < bytes; o += 64) { __builtin_prefetch((const char *)(g->t_out.pos + ro) + o); __builtin_prefetch((const char *)(g->t_out.neg + ro) + o); }
                __builtin_prefetch(g->t_out.scale + (size_t)gid * g->t_out.ng);
                if (k + 1 < K) {
                    const int nx = t * per_tree + cur[t]; const size_t ri = (size_t)nx * g->t_in.nch;
                    for (size_t o = 0; o < bytes; o += 64) { __builtin_prefetch((const char *)(g->t_in.pos + ri) + o); __builtin_prefetch((const char *)(g->t_in.neg + ri) + o); }
                    __builtin_prefetch(g->t_in.scale + (size_t)nx * g->t_in.ng); __builtin_prefetch(g->node_bias + nx);
                }
            }
        }
    if (N == 0) { for (int i = 0; i < S; i++) g->sc[i] = 0; goto coef; }
    // 2. the decay each path node missed since its last visit: one vectorised exp over the path
    for (int i = 0; i < S; i++) g->sd[i] = g->visited[g->sg[i]] ? g->clock[g->sq[i]] - g->stamp[g->sg[i]] : 0.0f;
    exp_array(g->sd, g->sf, S);
    // 3. read each node's state, then write to it
#ifdef __AVX512F__
    if (N % 16 == 0 && N <= 64 && !g->read_state) { // states as whole registers: scale, dot with C, add the key
        const int nc = N / 16;
        __m512 Cv[4], Bv[4];
        for (int c = 0; c < nc; c++) { Cv[c] = _mm512_loadu_ps(C + 16 * c); Bv[c] = _mm512_loadu_ps(B + 16 * c); }
        for (int i = 0; i < S; i++) {
            const int gid = g->sg[i];
            float *h = g->h + (size_t)gid * N, ctx = 0;
            const float w = g->write_logit ? dts[g->sq[i]] * g->sl[i] : dts[g->sq[i]];
            const __m512 wv = _mm512_set1_ps(w);
            if (g->visited[gid]) {
                const __m512 fv = _mm512_set1_ps(g->sf[i]);
                __m512 acc = _mm512_setzero_ps();
                for (int c = 0; c < nc; c++) {
                    const __m512 hv = _mm512_mul_ps(_mm512_loadu_ps(h + 16 * c), fv);
                    acc = _mm512_fmadd_ps(hv, Cv[c], acc);
                    _mm512_storeu_ps(h + 16 * c, _mm512_fmadd_ps(Bv[c], wv, hv));
                }
                ctx = _mm512_reduce_add_ps(acc);
            } else {
                for (int c = 0; c < nc; c++) _mm512_storeu_ps(h + 16 * c, _mm512_mul_ps(Bv[c], wv));
                g->visited[gid] = 1;
            }
            g->stamp[gid] = g->clock[g->sq[i]];
            g->sc[i] = ctx;
        }
    } else
#endif
    for (int i = 0; i < S; i++) {
        const int gid = g->sg[i];
        float *h = g->h + (size_t)gid * N, ctx = 0;
        if (g->visited[gid]) {
            const float f = g->sf[i];
            for (int n = 0; n < N; n++) { h[n] *= f; ctx += C[n] * h[n]; }
        } else {
            for (int n = 0; n < N; n++) h[n] = 0;
            g->visited[gid] = 1;
        }
        if (g->read_state) memcpy(g->found + i * N, h, N * sizeof(float)); // what this token found here
        const float w = g->write_logit ? dts[g->sq[i]] * g->sl[i] : dts[g->sq[i]];
        for (int n = 0; n < N; n++) h[n] += w * B[n];
        g->stamp[gid] = g->clock[g->sq[i]];
        g->sc[i] = ctx;
    }
coef:
    // 4. output coefficients, then one scaled add per path node
    if (g->act == 2) { gelu_array(g->sl, g->se, S); for (int i = 0; i < S; i++) g->sc[i] += g->se[i]; }
    else { for (int i = 0; i < S; i++) g->sc[i] += g->sl[i]; if (g->act == 0) { gelu_array(g->sc, g->se, S); memcpy(g->sc, g->se, S * sizeof(float)); } }
    memset(out, 0, d * sizeof(float));
#if HAVE_I8
    if (i8 && g_i16 && d % 32 == 0 && g->t_out.gch % 2 == 0) tm_out_i16(&g->t_out, g->sg, g->sc, S, out); else
#endif
    if (g->packed) tm_out_rows(&g->t_out, g->sg, g->sc, S, out);
    else for (int i = 0; i < S; i++) axpy(g->sc[i], g->node_out + (size_t)g->sg[i] * d, out, d);
    if (g->read_state) { // one dense read of the path's node states
        const int nr = g->n_trees * K * N;
        rmsnorm_eps(g->found, g->read_norm_w, g->found, nr, 1e-8f);
        for (int i = 0; i < d; i++) out[i] += dot(g->read_w + (size_t)i * nr, g->found, nr);
    }
}

// --------------------------------------------------------------------------------- Mamba-2

typedef struct {
    int d, di, N, H, P, kc, n_in, conv_dim;
    float *in_w, *conv_w, *conv_b, *dt_bias, *A, *D, *norm_w, *out_w, *conv_state, *state, *zx, *y, *tmp, *conv_t, *cv, *uq; TMat t_in, t_out; CMat c_in, c_out; int packed, colform, act_bits; signed char *q8;
} MLayer;

static MLayer *m2_load(int d, int di, int N, int H, int P, int kc, int act_bits) {
    MLayer *m = (MLayer *)xalloc(sizeof(MLayer));
    m->d = d; m->di = di; m->N = N; m->H = H; m->P = P; m->kc = kc; m->n_in = 2 * di + 2 * N + H; m->conv_dim = di + 2 * N;
    m->in_w = rdf((size_t)m->n_in * d); m->conv_w = rdf(m->conv_dim * kc); m->conv_b = rdf(m->conv_dim);
    m->dt_bias = rdf(H); m->A = rdf(H); m->D = rdf(H); m->norm_w = rdf(di); m->out_w = rdf((size_t)d * di);
    for (int h = 0; h < H; h++) m->A[h] = -expf(m->A[h]);
    m->conv_state = (float *)xalloc(m->conv_dim * kc * sizeof(float)); m->state = (float *)xalloc((size_t)di * N * sizeof(float));
    m->zx = (float *)xalloc(m->n_in * sizeof(float)); m->y = (float *)xalloc(di * sizeof(float)); m->tmp = (float *)xalloc(m->n_in * sizeof(float));
    m->packed = tm_pack(&m->t_in, m->in_w, m->n_in, d) && tm_pack(&m->t_out, m->out_w, d, di);
    m->colform = m->packed && g_colform && !act_bits && cm_from_tm(&m->c_in, &m->t_in) && cm_from_tm(&m->c_out, &m->t_out);
    m->act_bits = act_bits; m->uq = (float *)xalloc((d > di ? d : di) * sizeof(float)); m->q8 = (signed char *)xalloc((d > di ? d : di) + 64);
    m->conv_t = (float *)xalloc((size_t)kc * m->conv_dim * sizeof(float)); m->cv = (float *)xalloc(m->conv_dim * sizeof(float));
    for (int c = 0; c < m->conv_dim; c++) for (int j = 0; j < kc; j++) m->conv_t[(size_t)j * m->conv_dim + c] = m->conv_w[c * kc + j];
    return m;
}

static void m2_reset(MLayer *m) { memset(m->conv_state, 0, m->conv_dim * m->kc * sizeof(float)); memset(m->state, 0, (size_t)m->di * m->N * sizeof(float)); }

// Mamba2.step(): in_proj outputs are [z, x, B, C, dt].
static void m2_step(MLayer *m, const float *u, float *out) {
    const int d = m->d, di = m->di, N = m->N, P = m->P, kc = m->kc;
    __m512 xv[256];
    int i8 = 0;
#if HAVE_I8
    __m512i xq[64];
    if (m->act_bits == 8 && m->packed && d % 64 == 0 && di % 64 == 0 && di <= 4096 && m->t_in.gch % 4 == 0 && m->t_out.gch % 4 == 0) {
        i8 = 1;
        const float step = quant8(u, m->q8, d);
        for (int c = 0; c < d / 64; c++) xq[c] = _mm512_loadu_si512(m->q8 + 64 * c);
        for (int i = 0; i < m->n_in; i++) m->zx[i] = step * tm_dot_i8(&m->t_in, i, xq);
    }
#endif
    if (m->act_bits && !i8) { quant_float(u, m->uq, d, m->act_bits); u = m->uq; }
    if (i8) {}
    else if (m->colform) cm_matvec(&m->c_in, u, m->zx);
    else if (m->packed) { tm_load(u, xv, d / 16); for (int i = 0; i < m->n_in; i++) m->zx[i] = tm_dot(&m->t_in, i, xv); }
    else for (int i = 0; i < m->n_in; i++) m->zx[i] = dot(m->in_w + (size_t)i * d, u, d);
    float *z = m->zx, *xBC = m->zx + di, *dt = m->zx + di + m->conv_dim;
    { // causal depthwise conv; conv_state holds the previous kc - 1 inputs, oldest first
        const int cd = m->conv_dim; float *v = m->cv, *hist = m->conv_state;
        const float *wl = m->conv_t + (size_t)(kc - 1) * cd;
        for (int c = 0; c < cd; c++) v[c] = wl[c] * xBC[c] + m->conv_b[c];
        for (int j = 0; j < kc - 1; j++) { const float *wj = m->conv_t + (size_t)j * cd, *hj = hist + (size_t)j * cd; for (int c = 0; c < cd; c++) v[c] += wj[c] * hj[c]; }
        if (kc > 1) { memmove(hist, hist + cd, (size_t)(kc - 2) * cd * sizeof(float)); memcpy(hist + (size_t)(kc - 2) * cd, xBC, cd * sizeof(float)); }
        memcpy(xBC, v, cd * sizeof(float));
    }
    silu_array(xBC, m->tmp, m->conv_dim);
    const float *x = xBC, *B = xBC + di, *C = xBC + di + N;
    for (int h = 0; h < m->H; h++) {
        const float dth = softplusf(dt[h] + m->dt_bias[h]), dA = expf(dth * m->A[h]);
        for (int p = 0; p < P; p++) {
            const float xv = x[h * P + p], coef = dth * xv;
            float *s = m->state + (size_t)(h * P + p) * N, acc = 0;
            for (int n = 0; n < N; n++) { s[n] = s[n] * dA + coef * B[n]; acc += s[n] * C[n]; }
            m->y[h * P + p] = acc + m->D[h] * xv;
        }
    }
    float ss = 0;
    silu_array(z, m->tmp, di);
    for (int i = 0; i < di; i++) { m->y[i] *= z[i]; ss += m->y[i] * m->y[i]; }
    ss = 1.0f / sqrtf(ss / di + 1e-5f);
    for (int i = 0; i < di; i++) m->y[i] *= ss * m->norm_w[i];
#if HAVE_I8
    if (i8) {
        const float step = quant8(m->y, m->q8, di);
        for (int c = 0; c < di / 64; c++) xq[c] = _mm512_loadu_si512(m->q8 + 64 * c);
        for (int i = 0; i < d; i++) out[i] = step * tm_dot_i8(&m->t_out, i, xq);
        return;
    }
#endif
    if (m->act_bits) quant_float(m->y, m->y, di, m->act_bits);
    if (m->colform) cm_matvec(&m->c_out, m->y, out);
    else if (m->packed) { tm_load(m->y, xv, di / 16); for (int i = 0; i < d; i++) out[i] = tm_dot(&m->t_out, i, xv); }
    else for (int i = 0; i < d; i++) out[i] = dot(m->out_w + (size_t)i * di, m->y, di);
}

// ----------------------------------------------------------------------------------- model

// arch: 0 = GTS (first file format), 1 = Mamba-2, 2 = GTS, 3 = GTS with a Mamba-2 trunk (layers2) beside each tree
#define IS_M2(a) ((a) == 1 || (a) == 5)
typedef struct { int arch, V, d, n_layer; float *emb, *head_bias, *normf_w, **norm_w; void **layers, **layers2; float *u, *xn, *o, *o2; double t_mix; } Model;

static void model_reset(Model *M) { for (int l = 0; l < M->n_layer; l++) { if (!IS_M2(M->arch)) gts_reset((GLayer *)M->layers[l]); else m2_reset((MLayer *)M->layers[l]); if (M->arch == 3) m2_reset((MLayer *)M->layers2[l]); if (M->arch == 6) gts_reset((GLayer *)M->layers2[l]); } }

static void model_step(Model *M, int tok, float *logits) {
    const int d = M->d;
    memcpy(M->u, M->emb + (size_t)tok * d, d * sizeof(float));
    double t0 = now();
    for (int l = 0; l < M->n_layer; l++) {
        rmsnorm(M->u, M->norm_w[l], M->xn, d);
        if (!IS_M2(M->arch)) gts_step((GLayer *)M->layers[l], M->xn, M->o); else m2_step((MLayer *)M->layers[l], M->xn, M->o);
        if (M->arch == 3) { m2_step((MLayer *)M->layers2[l], M->xn, M->o2); for (int i = 0; i < d; i++) M->o[i] += M->o2[i]; }
        if (M->arch == 6) { gts_step((GLayer *)M->layers2[l], M->xn, M->o2); for (int i = 0; i < d; i++) M->o[i] += M->o2[i]; }
        for (int i = 0; i < d; i++) M->u[i] += M->o[i];
    }
    M->t_mix += now() - t0;
    rmsnorm(M->u, M->normf_w, M->xn, d);
    for (int v = 0; v < M->V; v++) logits[v] = dot(M->emb + (size_t)v * d, M->xn, d) + M->head_bias[v];
}

// --------------------------------------------------------------------------- synthetic models
// Models too large to build in PyTorch on a small machine: random packed ternary weights made directly, to time
// the layer stack. Same step functions as above; nothing here is checked against PyTorch, and nothing is trained.
//   ar_bench synth mixed  <width> <layers> <bank_trees> <bank_heads> <bank_state> <deep_trees> <deep_depth> <tokens>
//   ar_bench synth mamba2 <width> <layers> <d_state> <headdim> <act_bits> <tokens>
#ifdef __AVX512F__
static unsigned long long g_rng = 88172645463325252ULL;
static inline unsigned long long rnd64(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static inline float urand(void) { return (rnd64() >> 40) * (1.0f / 16777216.0f); }
static float *frand(size_t n, float lo, float hi) { float *p = (float *)xalloc((n + 1) * sizeof(float)); for (size_t i = 0; i < n; i++) p[i] = lo + (hi - lo) * urand(); return p; }
static void tm_random(TMat *m, int rows, int cols) { // about a quarter zeros, one scale per 128 weights
    int gsz = cols < 128 ? cols : 128;
    while (cols % gsz) gsz--;
    m->rows = rows; m->cols = cols; m->nch = cols / 16; m->gch = gsz / 16; m->ng = cols / gsz;
    const size_t n = (size_t)rows * m->nch;
    m->pos = (unsigned short *)xalloc(n * 2 + 64); m->neg = (unsigned short *)xalloc(n * 2 + 64);
    for (size_t i = 0; i < n; i++) { const unsigned long long r = rnd64(); const unsigned short nz = (unsigned short)(r | (r >> 16)), sg = (unsigned short)(r >> 32); m->pos[i] = nz & sg; m->neg[i] = nz & (unsigned short)~sg; }
    m->scale = (float *)xalloc(((size_t)rows * m->ng + 1) * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * m->ng; i++) m->scale[i] = (0.5f + urand()) / sqrtf((float)cols);
}
static GLayer *gts_synth(int d, int depth, int N, int n_trees, int n_heads, int act, int use_ctx) {
    GLayer *g = (GLayer *)xalloc(sizeof(GLayer));
    const int kc = 3;
    if (!use_ctx) N = 0;
    g->d = d; g->K = depth + 1; g->N = N; g->kc = kc; g->n_trees = n_trees; g->act = act; g->n_heads = n_heads;
    g->G = use_ctx ? n_heads * (depth + 1) : 0; g->n_nodes = n_trees * ((1 << (depth + 1)) - 1);
    g->act_bits = 8; g->write_logit = 1; g->read_state = 0;
    g->node_bias = frand(g->n_nodes, -0.03f, 0.03f);
    g->conv_t = frand((size_t)kc * d, -0.1f, 0.1f); for (int c = 0; c < d; c++) g->conv_t[(size_t)(kc - 1) * d + c] = 1.0f;
    g->conv_b = (float *)xalloc(d * sizeof(float)); g->xq = (signed char *)xalloc(d + 64);
    g->dt_bias = frand(g->G, -4.0f, -2.0f); g->A = frand(g->G, -16.0f, -1.0f);
    const int S = n_trees * g->K, sz = ((S > g->G ? S : g->G) + 16) * sizeof(float);
    g->found = (float *)xalloc(64);
    g->sl = (float *)xalloc(sz); g->sd = (float *)xalloc(sz); g->sf = (float *)xalloc(sz); g->sc = (float *)xalloc(sz); g->se = (float *)xalloc(sz);
    g->sg = (int *)xalloc(S * sizeof(int)); g->sq = (int *)xalloc(S * sizeof(int));
    for (int t = 0; t < n_trees; t++) for (int k = 0; k < g->K; k++) g->sq[t * g->K + k] = (t * n_heads / n_trees) * g->K + k;
    g->h = (float *)xalloc((size_t)g->n_nodes * N * sizeof(float)); g->stamp = (float *)xalloc((size_t)g->n_nodes * sizeof(float));
    g->visited = (unsigned char *)xalloc(g->n_nodes);
    g->hist = (float *)xalloc((size_t)(kc - 1) * d * sizeof(float)); g->x = (float *)xalloc(d * sizeof(float)); g->p = (float *)xalloc((2 * N + g->G + 1) * sizeof(float));
    tm_random(&g->t_in, g->n_nodes, d); tm_random(&g->t_out, g->n_nodes, d); tm_random(&g->t_ctx, 2 * N + g->G, d);
    g->packed = 1;
    return g;
}
static MLayer *m2_synth(int d, int N, int P, int act_bits) {
    MLayer *m = (MLayer *)xalloc(sizeof(MLayer));
    const int kc = 4, di = 2 * d, H = di / P;
    m->d = d; m->di = di; m->N = N; m->H = H; m->P = P; m->kc = kc; m->n_in = 2 * di + 2 * N + H; m->conv_dim = di + 2 * N; m->act_bits = act_bits;
    m->conv_t = frand((size_t)kc * m->conv_dim, -0.5f, 0.5f); m->conv_b = frand(m->conv_dim, -0.5f, 0.5f);
    m->dt_bias = frand(H, -4.0f, -2.0f); m->A = frand(H, -16.0f, -1.0f); m->D = frand(H, 1.0f, 1.0f); m->norm_w = frand(di, 1.0f, 1.0f);
    m->conv_state = (float *)xalloc((size_t)m->conv_dim * kc * sizeof(float)); m->state = (float *)xalloc((size_t)di * N * sizeof(float));
    m->zx = (float *)xalloc(m->n_in * sizeof(float)); m->y = (float *)xalloc(di * sizeof(float)); m->tmp = (float *)xalloc(m->n_in * sizeof(float));
    m->cv = (float *)xalloc(m->conv_dim * sizeof(float)); m->uq = (float *)xalloc(di * sizeof(float)); m->q8 = (signed char *)xalloc(di + 64);
    tm_random(&m->t_in, m->n_in, d); tm_random(&m->t_out, d, di);
    m->packed = 1;
    m->colform = g_colform && !act_bits && cm_from_tm(&m->c_in, &m->t_in) && cm_from_tm(&m->c_out, &m->t_out);
    return m;
}
static int synth_main(int argc, char **argv) {
    if (getenv("M2ROWS")) g_colform = 0;
    if (argc < 4) { fprintf(stderr, "see the comment above synth_main for usage\n"); return 1; }
    const int mixed = !strcmp(argv[2], "mixed"), d = atoi(argv[3]), L = atoi(argv[4]);
    Model M; memset(&M, 0, sizeof(M));
    M.d = d; M.n_layer = L; M.arch = mixed ? 6 : 1;
    M.layers = (void **)xalloc(L * sizeof(void *)); M.layers2 = (void **)xalloc(L * sizeof(void *)); M.norm_w = (float **)xalloc(L * sizeof(float *));
    double params = 0; long tokens;
    if (mixed) {
        if (argc < 11) return 1;
        const int bt = atoi(argv[5]), bh = atoi(argv[6]), bs = atoi(argv[7]), dt = atoi(argv[8]), dd = atoi(argv[9]); tokens = atol(argv[10]);
        for (int l = 0; l < L; l++) {
            M.layers[l] = gts_synth(d, 0, bs, bt, bh, 2, 1); M.layers2[l] = gts_synth(d, dd, 0, dt, 1, 0, 0); M.norm_w[l] = frand(d, 1.0f, 1.0f);
            const GLayer *a = (GLayer *)M.layers[l], *b = (GLayer *)M.layers2[l];
            params += (double)(a->n_nodes + b->n_nodes) * (2.0 * d + 1) + (double)(2 * bs + a->G) * d + 2.0 * a->G + 2.0 * 4 * d + d;
        }
        printf("mixed forest  width %d, %d layers; bank %d trees (%d heads, state %d); %d deep trees of depth %d; 8-bit activations\n", d, L, bt, bh, bs, dt, dd);
        printf("  path nodes per token per layer: %d of %d\n", bt + dt * (dd + 1), ((GLayer *)M.layers[0])->n_nodes + ((GLayer *)M.layers2[0])->n_nodes);
    } else {
        if (argc < 9) return 1;
        const int N = atoi(argv[5]), P = atoi(argv[6]), ab = atoi(argv[7]); tokens = atol(argv[8]);
        for (int l = 0; l < L; l++) {
            M.layers[l] = m2_synth(d, N, P, ab); M.norm_w[l] = frand(d, 1.0f, 1.0f);
            const MLayer *m = (MLayer *)M.layers[l];
            params += (double)m->n_in * d + (double)d * m->di + (double)m->conv_dim * 5 + 3.0 * m->H + m->di + d;
        }
        printf("Mamba-2       width %d, %d layers; expand 2, state %d, head width %d; %s activations\n", d, L, N, P, ab ? "8-bit" : "float");
    }
    printf("  mixer parameters: %.1fM  (packed ternary: about %.0f MB)\n", params / 1e6, params * 2 / 8 / 1e6);
    M.u = (float *)xalloc(d * sizeof(float)); M.xn = (float *)xalloc(d * sizeof(float)); M.o = (float *)xalloc(d * sizeof(float)); M.o2 = (float *)xalloc(d * sizeof(float));
    float *emb = frand((size_t)256 * d, -1.0f, 1.0f); // stand-in token vectors
    double sink = 0, best = 1e30, total = 0;
    for (int rep = 0; rep < 3; rep++) {
        model_reset(&M);
        const double t0 = now();
        for (long i = 0; i < tokens; i++) {
            memcpy(M.u, emb + (size_t)(rnd64() & 255) * d, d * sizeof(float));
            for (int l = 0; l < L; l++) {
                rmsnorm(M.u, M.norm_w[l], M.xn, d);
                if (mixed) { gts_step((GLayer *)M.layers[l], M.xn, M.o); gts_step((GLayer *)M.layers2[l], M.xn, M.o2); for (int c = 0; c < d; c++) M.o[c] += M.o2[c]; }
                else m2_step((MLayer *)M.layers[l], M.xn, M.o);
                for (int c = 0; c < d; c++) M.u[c] += M.o[c];
            }
            sink += M.u[0];
        }
        const double dt = (now() - t0) / tokens; total += dt; if (dt < best) best = dt;
    }
    printf("  one token through the stack: %.1f us (best of 3; mean %.1f)   = %.2f us per layer   128 tokens: %.1f ms   [%g]\n",
           best * 1e6, total / 3 * 1e6, best * 1e6 / L, best * 128 * 1e3, sink);
    return 0;
}
#else
static int synth_main(int argc, char **argv) { (void)argc; (void)argv; fprintf(stderr, "synthetic models need AVX-512\n"); return 1; }
#endif

int main(int argc, char **argv) {
    if (argc > 1 && !strcmp(argv[1], "synth")) return synth_main(argc, argv);
    if (argc < 2) { fprintf(stderr, "usage: ar_bench model.bin [n_tokens]\n"); return 1; }
    const long n_tokens = argc > 2 ? atol(argv[2]) : 100000;
    if (argc > 3 && !strcmp(argv[3], "float")) g_use_pack = 0; // run the ternary weights as plain float32
    if (argc > 3 && !strcmp(argv[3], "i16out")) g_i16 = 1;
    if (argc > 3 && !strcmp(argv[3], "m2rows")) g_colform = 0; // Mamba-2 float projections as row dot products (the earlier kernel)     // also sum the output rows with 16-bit integer coefficients (no faster, less exact)
    g_f = fopen(argv[1], "rb"); if (!g_f) { perror("open"); return 1; }
    Model M; memset(&M, 0, sizeof(M));
    M.arch = rdi(); M.V = rdi(); M.d = rdi(); M.n_layer = rdi();
    int dims[20] = {0, 0, 0, 0, 1, 1, 0, 1, 0}; const int n_dims = M.arch == 0 ? 3 : M.arch == 1 ? 5 : M.arch == 2 ? 8 : M.arch == 4 ? 9 : M.arch == 5 ? 6 : M.arch == 6 ? 20 : 13; // 6: two GTS mixers per layer // 4: GTS with act_bits, 5: Mamba-2 with act_bits // arch 0: GTS (first format), 1: Mamba-2, 2: GTS with flags
    for (int i = 0; i < n_dims; i++) dims[i] = rdi();
    M.emb = rdf((size_t)M.V * M.d); M.head_bias = rdf(M.V); M.normf_w = rdf(M.d);
    M.layers = (void **)xalloc(M.n_layer * sizeof(void *)); M.layers2 = (void **)xalloc(M.n_layer * sizeof(void *)); M.norm_w = (float **)xalloc(M.n_layer * sizeof(float *));
    for (int l = 0; l < M.n_layer; l++) {
        M.norm_w[l] = rdf(M.d);
        M.layers[l] = !IS_M2(M.arch) ? (void *)gts_load(M.d, dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], M.arch == 4 || M.arch == 6 ? dims[8] : 0, M.arch == 6 ? dims[9] : 1) : (void *)m2_load(M.d, dims[0], dims[1], dims[2], dims[3], dims[4], M.arch == 5 ? dims[5] : 0);
        if (M.arch == 3) M.layers2[l] = (void *)m2_load(M.d, dims[8], dims[9], dims[10], dims[11], dims[12], 0);
        if (M.arch == 6) M.layers2[l] = (void *)gts_load(M.d, dims[10], dims[11], dims[12], dims[13], dims[14], dims[15], dims[16], dims[17], dims[18], dims[19]);
    }
    const int T = rdi();
    int *ids = (int *)xalloc(T * sizeof(int)); if (fread(ids, sizeof(int), T, g_f) != (size_t)T) return 1;
    float *ref = rdf((size_t)T * M.V);
    fclose(g_f);
    M.u = (float *)xalloc(M.d * sizeof(float)); M.xn = (float *)xalloc(M.d * sizeof(float)); M.o = (float *)xalloc(M.d * sizeof(float)); M.o2 = (float *)xalloc(M.d * sizeof(float));
    float *logits = (float *)xalloc(M.V * sizeof(float));

    // 1. the kernel must reproduce PyTorch
    model_reset(&M);
    float worst = 0; int agree = 0; double nll_c = 0, nll_ref = 0; // next-token loss from this kernel's logits and from PyTorch's
    for (int t = 0; t < T; t++) {
        model_step(&M, ids[t], logits);
        int a = 0, b = 0;
        for (int v = 0; v < M.V; v++) {
            const float e = fabsf(logits[v] - ref[(size_t)t * M.V + v]); if (e > worst) worst = e;
            if (logits[v] > logits[a]) a = v;
            if (ref[(size_t)t * M.V + v] > ref[(size_t)t * M.V + b]) b = v;
        }
        agree += a == b;
        if (t + 1 < T) {
            double zc = 0, zr = 0; const float mc = logits[a], mr = ref[(size_t)t * M.V + b];
            for (int v = 0; v < M.V; v++) { zc += exp(logits[v] - mc); zr += exp(ref[(size_t)t * M.V + v] - mr); }
            nll_c += log(zc) + mc - logits[ids[t + 1]]; nll_ref += log(zr) + mr - ref[(size_t)t * M.V + ids[t + 1]];
        }
    }
    printf("%s  %d layers, width %d\n", IS_M2(M.arch) ? "Mamba-2" : M.arch == 3 ? "GTS + trunk" : M.arch == 6 ? "GTS mixed" : "GTS    ", M.n_layer, M.d);
    printf("  weights: %d tensors packed ternary, %d float32\n", g_packed, g_float);
    printf("  check: max |logit - PyTorch logit| = %.2e over %d tokens, same top-1 on %d; loss on them %.4f (PyTorch %.4f)\n", worst, T, agree, nll_c / (T - 1), nll_ref / (T - 1));

    // 2. timing: a continuous stream, state reset every T tokens
    double sink = 0; M.t_mix = 0;
    const double t0 = now();
    for (long i = 0; i < n_tokens; i++) {
        if (i % T == 0) model_reset(&M);
        model_step(&M, ids[i % T], logits);
        sink += logits[0];
    }
    const double dt = now() - t0;
    printf("  speed: %.1f us per token  (%.0f tokens/s), of which mixers %.1f us  = %.2f us per layer  [%g]\n",
           dt / n_tokens * 1e6, n_tokens / dt, M.t_mix / n_tokens * 1e6, M.t_mix / n_tokens * 1e6 / M.n_layer, sink);
    return 0;
}
