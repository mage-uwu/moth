// Golden Tree Snake (GTS) fork, 2026.
// Float32 CPU kernel for one GTS mixer (the algorithm of GTS.forward_reference), a float32
// Mamba-2 single-token step for comparison, and a benchmark driver.
//
//   gcc -O3 -march=native -ffast-math -funroll-loops gts_kernel.c -o gts_kernel -lm
//   ./gts_kernel verify in.bin out.bin      run one GTS layer on exported weights
//   ./gts_kernel bench [uniform|zipf]       time GTS against the Mamba-2 step
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static uint64_t rng_s = 88172645463325252ULL;
static inline uint64_t rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return rng_s; }
static inline float urand(void) { return (rnd() >> 40) * (1.0f / 16777216.0f); }
static inline float nrand(void) { return (urand() + urand() + urand() + urand() - 2.0f) * 1.7320508f; }
static void *xalloc(size_t bytes) { void *p; if (posix_memalign(&p, 64, bytes ? bytes : 64)) { fprintf(stderr, "oom\n"); exit(1); } return p; }
static float *falloc(size_t n) { return (float *)xalloc(n * sizeof(float)); }
static float *frand(size_t n, float k) { float *p = falloc(n); for (size_t i = 0; i < n; i++) p[i] = (2 * urand() - 1) * k; return p; }

static inline float dot(const float *a, const float *b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static inline void axpy(float alpha, const float *x, float *y, int n) { for (int i = 0; i < n; i++) y[i] += alpha * x[i]; }
static inline float softplusf(float x) { return x > 20.0f ? x : log1pf(expf(x)); }
static inline float geluf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678f)); }
static inline float siluf(float x) { return x / (1.0f + expf(-x)); }
static void rmsnorm(const float *x, float *y, int n) { float s = 0; for (int i = 0; i < n; i++) s += x[i] * x[i]; s = 1.0f / sqrtf(s / n + 1e-5f); for (int i = 0; i < n; i++) y[i] = x[i] * s; }

// ----------------------------------------------------------------------------------- GTS

typedef struct {
    int d, K, N, n_nodes, max_len;
    float *node_in, *node_bias, *node_out; // n_nodes*d, n_nodes, n_nodes*d
    float *conv_w, *conv_b;                // d*3, d
    float *ctx_w;                          // (3N+K)*d, rows ordered [B, C_fwd, C_bwd, dt]
    float *dt_bias, *A;                    // K, K   (A = -exp(A_log))
    float *h, *stamp; int *epoch; int cur_epoch; // node states: n_nodes*N, n_nodes, n_nodes
    float *x, *B, *Cf, *Cb, *dt, *a, *logit, *ctx; int *nodes; // per-sequence workspace
    double t_proj, t_walk, t_state, t_out;
} GTS;

static void gts_alloc_work(GTS *g, int max_len) {
    g->max_len = max_len;
    g->h = falloc((size_t)g->n_nodes * g->N); g->stamp = falloc(g->n_nodes);
    g->epoch = (int *)xalloc(g->n_nodes * sizeof(int)); memset(g->epoch, 0, g->n_nodes * sizeof(int)); g->cur_epoch = 0;
    g->x = falloc((size_t)max_len * g->d);
    g->B = falloc(max_len * g->N); g->Cf = falloc(max_len * g->N); g->Cb = falloc(max_len * g->N);
    g->dt = falloc(max_len * g->K); g->a = falloc(max_len * g->K); g->logit = falloc(max_len * g->K); g->ctx = falloc(max_len * g->K);
    g->nodes = (int *)xalloc(max_len * g->K * sizeof(int));
}

// One direction of context. Nodes are only touched when visited; the decay they missed is applied on the visit.
static void gts_scan(GTS *g, int L, int reverse, const float *C) {
    const int K = g->K, N = g->N;
    float clock[64] = {0};
    const int ep = ++g->cur_epoch; // nothing is cleared between scans: a stale epoch means "empty"
    for (int i = 0; i < L; i++) {
        const int t = reverse ? L - 1 - i : i;
        const float *Ct = C + t * N, *Bt = g->B + t * N;
        for (int k = 0; k < K; k++) {
            clock[k] += g->a[t * K + k];
            const int gid = g->nodes[t * K + k];
            float *h = g->h + (size_t)gid * N;
            if (g->epoch[gid] == ep) {
                const float f = expf(clock[k] - g->stamp[gid]);
                float c = 0;
                for (int n = 0; n < N; n++) { h[n] *= f; c += Ct[n] * h[n]; }
                g->ctx[t * K + k] += c;
            } else {
                for (int n = 0; n < N; n++) h[n] = 0;
                g->epoch[gid] = ep;
            }
            const float s = g->dt[t * K + k] * g->logit[t * K + k];
            for (int n = 0; n < N; n++) h[n] += s * Bt[n];
            g->stamp[gid] = clock[k];
        }
    }
}

// u, out: L*d. u is the (normalised) input to the mixer; out is the mixer output.
static void gts_forward(GTS *g, const float *u, int L, float *out) {
    const int d = g->d, K = g->K, N = g->N;
    double t0 = now();
    for (int t = 0; t < L; t++) { // centred depthwise conv
        const float *um = t > 0 ? u + (size_t)(t - 1) * d : NULL, *uc = u + (size_t)t * d, *up = t < L - 1 ? u + (size_t)(t + 1) * d : NULL;
        float *x = g->x + (size_t)t * d;
        for (int c = 0; c < d; c++) {
            float v = g->conv_w[c * 3 + 1] * uc[c] + g->conv_b[c];
            if (um) v += g->conv_w[c * 3] * um[c];
            if (up) v += g->conv_w[c * 3 + 2] * up[c];
            x[c] = v;
        }
    }
    for (int t = 0; t < L; t++) { // key, queries, step sizes
        const float *x = g->x + (size_t)t * d;
        for (int n = 0; n < N; n++) {
            g->B[t * N + n] = dot(g->ctx_w + (size_t)n * d, x, d);
            g->Cf[t * N + n] = dot(g->ctx_w + (size_t)(N + n) * d, x, d);
            g->Cb[t * N + n] = dot(g->ctx_w + (size_t)(2 * N + n) * d, x, d);
        }
        for (int k = 0; k < K; k++) {
            const float dt = softplusf(dot(g->ctx_w + (size_t)(3 * N + k) * d, x, d) + g->dt_bias[k]);
            g->dt[t * K + k] = dt; g->a[t * K + k] = dt * g->A[k];
        }
    }
    double t1 = now();
    for (int t = 0; t < L; t++) { // tree walk
        const float *x = g->x + (size_t)t * d;
        int cur = 0;
        for (int k = 0; k < K; k++) {
            if (k < K - 1) { // the two children are adjacent rows: fetch the start of both while this dot runs
                const char *child = (const char *)(g->node_in + (size_t)(2 * cur + 1) * d);
                for (int o = 0; o < 6144; o += 512) __builtin_prefetch(child + o);
            }
            const float lg = dot(x, g->node_in + (size_t)cur * d, d) + g->node_bias[cur];
            g->nodes[t * K + k] = cur; g->logit[t * K + k] = lg; g->ctx[t * K + k] = 0;
            __builtin_prefetch(g->node_out + (size_t)cur * d);
            cur = 2 * cur + 1 + (lg > 0);
        }
    }
    double t2 = now();
    gts_scan(g, L, 0, g->Cf);
    gts_scan(g, L, 1, g->Cb);
    double t3 = now();
    for (int t = 0; t < L; t++) { // read vectors
        float *o = out + (size_t)t * d;
        memset(o, 0, d * sizeof(float));
        for (int k = 0; k < K; k++)
            axpy(geluf(g->logit[t * K + k] + g->ctx[t * K + k]), g->node_out + (size_t)g->nodes[t * K + k] * d, o, d);
    }
    double t4 = now();
    g->t_proj += t1 - t0; g->t_walk += t2 - t1; g->t_state += t3 - t2; g->t_out += t4 - t3;
}

static GTS *gts_random(int d, int depth, int N, int max_len) {
    GTS *g = (GTS *)calloc(1, sizeof(GTS));
    g->d = d; g->K = depth + 1; g->N = N; g->n_nodes = (1 << (depth + 1)) - 1;
    const float k_in = 1.0f / sqrtf(d), k_out = 1.0f / sqrtf(g->K);
    g->node_in = frand((size_t)g->n_nodes * d, k_in); g->node_bias = frand(g->n_nodes, k_in); g->node_out = frand((size_t)g->n_nodes * d, k_out);
    g->conv_w = falloc(d * 3); g->conv_b = falloc(d);
    for (int c = 0; c < d; c++) { g->conv_w[c * 3] = 0.1f * nrand(); g->conv_w[c * 3 + 1] = 1; g->conv_w[c * 3 + 2] = 0.1f * nrand(); g->conv_b[c] = 0; }
    g->ctx_w = frand((size_t)(3 * N + g->K) * d, k_in);
    g->dt_bias = falloc(g->K); g->A = falloc(g->K);
    for (int k = 0; k < g->K; k++) { g->dt_bias[k] = -3.0f + urand(); g->A[k] = -(1.0f + 15.0f * urand()); }
    gts_alloc_work(g, max_len);
    return g;
}

// --------------------------------------------------------------------------------- Mamba-2

typedef struct {
    int d, d_inner, N, H, P, d_in_proj, conv_dim;
    float *in_w, *out_w, *conv_w, *conv_b, *conv_state, *dt_bias, *A, *D, *norm_w, *state;
    float *zx, *y;
    double t_proj, t_rest;
} Mamba2;

static Mamba2 *mamba2_random(int d) {
    Mamba2 *m = (Mamba2 *)calloc(1, sizeof(Mamba2));
    m->d = d; m->d_inner = 2 * d; m->N = 128; m->P = 64; m->H = m->d_inner / m->P;
    m->d_in_proj = 2 * m->d_inner + 2 * m->N + m->H; m->conv_dim = m->d_inner + 2 * m->N;
    m->in_w = frand((size_t)m->d_in_proj * d, 1.0f / sqrtf(d)); m->out_w = frand((size_t)d * m->d_inner, 1.0f / sqrtf(m->d_inner));
    m->conv_w = frand(m->conv_dim * 4, 0.5f); m->conv_b = frand(m->conv_dim, 0.5f);
    m->conv_state = falloc(m->conv_dim * 4); memset(m->conv_state, 0, m->conv_dim * 4 * sizeof(float));
    m->dt_bias = falloc(m->H); m->A = falloc(m->H); m->D = falloc(m->H); m->norm_w = falloc(m->d_inner);
    for (int h = 0; h < m->H; h++) { m->dt_bias[h] = -3.0f + urand(); m->A[h] = -(1.0f + 15.0f * urand()); m->D[h] = 1; }
    for (int i = 0; i < m->d_inner; i++) m->norm_w[i] = 1;
    m->state = falloc((size_t)m->d_inner * m->N); memset(m->state, 0, (size_t)m->d_inner * m->N * sizeof(float));
    m->zx = falloc(m->d_in_proj); m->y = falloc(m->d_inner);
    return m;
}

// Mamba2.step(): one token. Order of in_proj outputs: [z, x, B, C, dt].
static void mamba2_step(Mamba2 *m, const float *u, float *out) {
    const int d = m->d, di = m->d_inner, N = m->N, P = m->P;
    double t0 = now();
    for (int i = 0; i < m->d_in_proj; i++) m->zx[i] = dot(m->in_w + (size_t)i * d, u, d);
    double t1 = now();
    float *z = m->zx, *xBC = m->zx + di, *dt = m->zx + di + m->conv_dim;
    for (int c = 0; c < m->conv_dim; c++) { // causal depthwise conv over the last 4 tokens, then SiLU
        float *s = m->conv_state + c * 4;
        s[0] = s[1]; s[1] = s[2]; s[2] = s[3]; s[3] = xBC[c];
        const float *w = m->conv_w + c * 4;
        xBC[c] = siluf(s[0] * w[0] + s[1] * w[1] + s[2] * w[2] + s[3] * w[3] + m->conv_b[c]);
    }
    const float *x = xBC, *B = xBC + di, *C = xBC + di + N;
    for (int h = 0; h < m->H; h++) { // SSM step: state = state * dA + dt * x (outer) B ; y = state . C + D * x
        const float dth = softplusf(dt[h] + m->dt_bias[h]), dA = expf(dth * m->A[h]);
        for (int p = 0; p < P; p++) {
            const float xv = x[h * P + p], coef = dth * xv;
            float *s = m->state + (size_t)(h * P + p) * N, acc = 0;
            for (int n = 0; n < N; n++) { s[n] = s[n] * dA + coef * B[n]; acc += s[n] * C[n]; }
            m->y[h * P + p] = acc + m->D[h] * xv;
        }
    }
    float ss = 0; // gate, then RMSNorm
    for (int i = 0; i < di; i++) { m->y[i] *= siluf(z[i]); ss += m->y[i] * m->y[i]; }
    ss = 1.0f / sqrtf(ss / di + 1e-5f);
    for (int i = 0; i < di; i++) m->y[i] *= ss * m->norm_w[i];
    double t2 = now();
    for (int i = 0; i < d; i++) out[i] = dot(m->out_w + (size_t)i * di, m->y, di);
    double t3 = now();
    m->t_proj += (t1 - t0) + (t3 - t2); m->t_rest += t2 - t1;
}

// ------------------------------------------------------------------------------------ main

static void rd(void *p, size_t sz, size_t n, FILE *f) { if (fread(p, sz, n, f) != n) { fprintf(stderr, "short read\n"); exit(1); } }

static int verify(const char *in_path, const char *out_path) {
    FILE *f = fopen(in_path, "rb"); if (!f) return 1;
    int hdr[4]; rd(hdr, sizeof(int), 4, f);
    const int d = hdr[0], depth = hdr[1], N = hdr[2], L = hdr[3];
    GTS *g = (GTS *)calloc(1, sizeof(GTS));
    g->d = d; g->K = depth + 1; g->N = N; g->n_nodes = (1 << (depth + 1)) - 1;
    g->node_in = falloc((size_t)g->n_nodes * d); rd(g->node_in, 4, (size_t)g->n_nodes * d, f);
    g->node_bias = falloc(g->n_nodes); rd(g->node_bias, 4, g->n_nodes, f);
    g->node_out = falloc((size_t)g->n_nodes * d); rd(g->node_out, 4, (size_t)g->n_nodes * d, f);
    g->conv_w = falloc(d * 3); rd(g->conv_w, 4, d * 3, f);
    g->conv_b = falloc(d); rd(g->conv_b, 4, d, f);
    g->ctx_w = falloc((size_t)(3 * N + g->K) * d); rd(g->ctx_w, 4, (size_t)(3 * N + g->K) * d, f);
    g->dt_bias = falloc(g->K); rd(g->dt_bias, 4, g->K, f);
    g->A = falloc(g->K); rd(g->A, 4, g->K, f);
    for (int k = 0; k < g->K; k++) g->A[k] = -expf(g->A[k]); // file holds A_log
    float *u = falloc((size_t)L * d), *out = falloc((size_t)L * d); rd(u, 4, (size_t)L * d, f); fclose(f);
    gts_alloc_work(g, L);
    gts_forward(g, u, L, out);
    gts_forward(g, u, L, out); // twice: the second run must not see the first run's node states
    f = fopen(out_path, "wb"); fwrite(out, 4, (size_t)L * d, f); fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "verify")) return verify(argv[2], argv[3]);
    const int zipf = argc >= 3 && !strcmp(argv[2], "zipf");
    const int d = 768, depth = 11, N = 16, L = 128, V = 30522;
    const int gts_layers = 12, gts_seqs = 40, m2_layers = 12, m2_tokens = 768;

    float *emb = falloc((size_t)V * d); for (size_t i = 0; i < (size_t)V * d; i++) emb[i] = nrand();
    double *cdf = (double *)xalloc(V * sizeof(double)); double tot = 0;
    for (int i = 0; i < V; i++) { tot += zipf ? 1.0 / (i + 1) : 1.0; cdf[i] = tot; }
    int *ids = (int *)xalloc(gts_seqs * L * sizeof(int));
    for (int i = 0; i < gts_seqs * L; i++) { double r = urand() * tot; int lo = 0, hi = V - 1; while (lo < hi) { int mid = (lo + hi) / 2; if (cdf[mid] < r) lo = mid + 1; else hi = mid; } ids[i] = lo; }

    printf("tokens: %s over a %d-word vocabulary, width %d\n", zipf ? "zipf" : "uniform", V, d);

    // ---- GTS: 12 layers, 128-token sequences, both directions
    GTS **g = (GTS **)xalloc(gts_layers * sizeof(GTS *));
    for (int l = 0; l < gts_layers; l++) g[l] = gts_random(d, depth, N, L);
    float *u = falloc((size_t)L * d), *xn = falloc((size_t)L * d), *o = falloc((size_t)L * d);
    char *seen = (char *)calloc(g[0]->n_nodes, 1); double sink = 0;
    double t0 = now();
    for (int s = 0; s < gts_seqs; s++) {
        for (int t = 0; t < L; t++) memcpy(u + (size_t)t * d, emb + (size_t)ids[s * L + t] * d, d * sizeof(float));
        for (int l = 0; l < gts_layers; l++) {
            for (int t = 0; t < L; t++) rmsnorm(u + (size_t)t * d, xn + (size_t)t * d, d);
            gts_forward(g[l], xn, L, o);
            for (size_t i = 0; i < (size_t)L * d; i++) u[i] += o[i];
            if (l == gts_layers - 1) for (int i = 0; i < L * g[l]->K; i++) seen[g[l]->nodes[i]] = 1;
        }
        sink += u[0];
    }
    double tg = now() - t0;
    const double g_steps = (double)gts_seqs * L * gts_layers;
    double tp = 0, tw = 0, ts = 0, to = 0; for (int l = 0; l < gts_layers; l++) { tp += g[l]->t_proj; tw += g[l]->t_walk; ts += g[l]->t_state; to += g[l]->t_out; }
    int used = 0; for (int i = 0; i < g[0]->n_nodes; i++) used += seen[i];
    printf("GTS      %8.2f us per token per layer   (conv+ctx_proj %.2f, walk %.2f, states %.2f, read %.2f)\n", tg / g_steps * 1e6, tp / g_steps * 1e6, tw / g_steps * 1e6, ts / g_steps * 1e6, to / g_steps * 1e6);
    printf("         last layer visited %d of %d nodes over %d tokens\n", used, g[0]->n_nodes, gts_seqs * L);

    // ---- Mamba-2: 12 layers, one direction, token at a time
    Mamba2 **m = (Mamba2 **)xalloc(m2_layers * sizeof(Mamba2 *));
    for (int l = 0; l < m2_layers; l++) m[l] = mamba2_random(d);
    float *v = falloc(d), *vn = falloc(d), *vo = falloc(d);
    t0 = now();
    for (int t = 0; t < m2_tokens; t++) {
        memcpy(v, emb + (size_t)ids[t] * d, d * sizeof(float));
        for (int l = 0; l < m2_layers; l++) { rmsnorm(v, vn, d); mamba2_step(m[l], vn, vo); for (int i = 0; i < d; i++) v[i] += vo[i]; }
        sink += v[0];
    }
    double tm = now() - t0;
    const double m_steps = (double)m2_tokens * m2_layers;
    double mp = 0, mr = 0; for (int l = 0; l < m2_layers; l++) { mp += m[l]->t_proj; mr += m[l]->t_rest; }
    printf("Mamba-2  %8.2f us per token per layer   (in_proj+out_proj %.2f, conv+scan+norm %.2f)\n", tm / m_steps * 1e6, mp / m_steps * 1e6, mr / m_steps * 1e6);
    printf("ratio    %8.1fx   (checksum %g)\n", (tm / m_steps) / (tg / g_steps), sink);
    return 0;
}
