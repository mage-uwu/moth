// Golden Tree Snake (GTS) fork, 2026.
// Inference for the bidirectional mixed-forest masked LM (GTSForMaskedLM, mixer "mixed") on one CPU core.
// Loads the file scripts/export_encoder.py writes, checks its logits at the test's masked positions against PyTorch's,
// then times encoding whole sequences.
//
//   gcc -O3 -march=native -ffast-math -funroll-loops kernel/enc_bench.c -o kernel/enc_bench -lm
//   ./kernel/enc_bench enc.bin [repeats]
//
// Each layer: x += bank(norm x) + deep(norm x). Each of the two mixers has its own centred depthwise conv, and its
// input is quantised to int8 per token (as trained). The bank is depth-0 trees: every token visits every node, so
// its context is a forward pass over the sequence and a backward pass, each carrying one d_state vector per node and
// decaying it by the token's per-head factor before reading (the token's own write is excluded). The deep trees are
// stateless: each token walks one root-to-leaf path per tree. Only visited nodes' rows are read: ternary weights
// packed two bits per weight, integer dot products over the int8 input (VNNI when present), output rows summed with
// masked adds. Reuses ar_bench.c's packed-ternary routines.
#define main ar_bench_main
#include "ar_bench.c"
#undef main

#if !HAVE_I8
#error "enc_bench needs AVX-512 (F and BW)"
#endif

typedef struct {
    int n_trees, depth, K, per_tree, n_nodes, N, H, G, ctx;  // G: clocks (heads); ctx: rows of ctx_proj
    TMat t_in, t_out, t_ctx;
    float *bias, *conv_t, *conv_b, *dt_bias, *A;
} Mix;

typedef struct { float *norm; Mix bank, deep; } Layer;

static int V, D, NL, KC, TB, HB, NB, TD, DD;
static float *emb, *head_bias, *norm_f;
static Layer *layers;

static void load_mix(Mix *m, int n_trees, int depth, int N, int H, int ctx) {
    m->n_trees = n_trees; m->depth = depth; m->K = depth + 1; m->per_tree = (1 << (depth + 1)) - 1;
    m->n_nodes = n_trees * m->per_tree; m->N = N; m->H = H; m->G = ctx ? H : 0;
    float *w_in = rdf((size_t)m->n_nodes * D); m->bias = rdf(m->n_nodes); float *w_out = rdf((size_t)m->n_nodes * D);
    float *cw = rdf((size_t)D * KC); m->conv_b = rdf(D);
    m->conv_t = (float *)xalloc((size_t)KC * D * sizeof(float));
    for (int c = 0; c < D; c++) for (int j = 0; j < KC; j++) m->conv_t[(size_t)j * D + c] = cw[c * KC + j];
    int ok = tm_pack(&m->t_in, w_in, m->n_nodes, D) && tm_pack(&m->t_out, w_out, m->n_nodes, D);
    if (ctx) {
        m->ctx = 3 * N + m->G;  // B, C_fwd, C_bwd, dt
        float *w_ctx = rdf((size_t)m->ctx * D);
        m->dt_bias = rdf(m->G); m->A = rdf(m->G);
        for (int h = 0; h < m->G; h++) m->A[h] = -expf(m->A[h]);
        ok = ok && tm_pack(&m->t_ctx, w_ctx, m->ctx, D);
        free(w_ctx);
    }
    if (!ok) { fprintf(stderr, "a ternary tensor did not pack\n"); exit(1); }
    free(w_in); free(w_out); free(cw);
}

// Scratch, sized for the longest sequence.
static float *X, *U, *XC, *OUT, *P, *LB, *SRC, *CTX, *AD, *STEP, *HST;
static signed char *Q;

static void alloc_scratch(int T) {
    X = (float *)xalloc((size_t)T * D * 4); U = (float *)xalloc((size_t)T * D * 4); OUT = (float *)xalloc((size_t)T * D * 4);
    XC = (float *)xalloc((size_t)D * 4); Q = (signed char *)xalloc((size_t)T * (D + 64));
    STEP = (float *)xalloc((size_t)T * 4);
    P = (float *)xalloc((size_t)T * (3 * NB + HB) * 4); LB = (float *)xalloc((size_t)T * TB * 4);
    SRC = (float *)xalloc((size_t)T * TB * 4); CTX = (float *)xalloc((size_t)T * TB * 4); AD = (float *)xalloc((size_t)T * HB * 4);
    HST = (float *)xalloc((size_t)TB * NB * 4);
}

// The mixer's input for every token: centred conv over U, then int8 per token (codes in Q, step in STEP).
static void conv_quant(const Mix *m, int T) {
    const int pad = KC / 2;
    for (int t = 0; t < T; t++) {
        for (int c = 0; c < D; c++) XC[c] = m->conv_b[c];
        for (int j = 0; j < KC; j++) {
            const int s = t + j - pad;
            if (s < 0 || s >= T) continue;
            const float *w = m->conv_t + (size_t)j * D, *u = U + (size_t)s * D;
            for (int c = 0; c < D; c++) XC[c] += w[c] * u[c];
        }
        STEP[t] = quant8(XC, Q + (size_t)t * (D + 64), D);
    }
}

static inline void load_q(int t, __m512i *xq) { for (int c = 0; c < D / 64; c++) xq[c] = _mm512_loadu_si512(Q + (size_t)t * (D + 64) + 64 * c); }

static void bank(const Mix *m, int T) {
    const int N = m->N, H = m->H, nc = m->ctx, per_head = m->n_trees / H;
    __m512i xq[16];
    conv_quant(m, T);
    for (int t = 0; t < T; t++) {  // per token: key, queries, step sizes, the 32 nodes' logits
        load_q(t, xq);
        float *p = P + (size_t)t * nc, *lg = LB + (size_t)t * m->n_trees, dts[64], tmp[64];
        for (int r = 0; r < nc; r++) p[r] = STEP[t] * tm_dot_i8(&m->t_ctx, r, xq);
        for (int i = 0; i < m->n_trees; i++) lg[i] = STEP[t] * tm_dot_i8(&m->t_in, i, xq) + m->bias[i];
        for (int h = 0; h < H; h++) tmp[h] = p[3 * N + h] + m->dt_bias[h];
        softplus_array(tmp, dts, tmp + 32, H);
        for (int h = 0; h < H; h++) AD[(size_t)t * H + h] = expf(dts[h] * m->A[h]);  // the token's decay factor per head
        for (int i = 0; i < m->n_trees; i++) SRC[(size_t)t * m->n_trees + i] = dts[i / per_head] * lg[i];  // what it writes
        for (int i = 0; i < m->n_trees; i++) CTX[(size_t)t * m->n_trees + i] = 0.0f;
    }
    for (int dir = 0; dir < 2; dir++) {  // forward with C_fwd, then backward with C_bwd
        memset(HST, 0, (size_t)m->n_trees * N * 4);
        for (int k = 0; k < T; k++) {
            const int t = dir ? T - 1 - k : k;
            const float *p = P + (size_t)t * nc, *B = p, *C = p + (1 + dir) * N, *ad = AD + (size_t)t * H;
            const __m512 Bv = _mm512_loadu_ps(B), Cv = _mm512_loadu_ps(C);  // N = 16: one register each
            for (int i = 0; i < m->n_trees; i++) {
                float *h = HST + (size_t)i * N;
                __m512 hv = _mm512_mul_ps(_mm512_loadu_ps(h), _mm512_set1_ps(ad[i / per_head]));  // decay, then read
                CTX[(size_t)t * m->n_trees + i] += _mm512_reduce_add_ps(_mm512_mul_ps(hv, Cv));
                hv = _mm512_fmadd_ps(_mm512_set1_ps(SRC[(size_t)t * m->n_trees + i]), Bv, hv);  // then write
                _mm512_storeu_ps(h, hv);
            }
        }
    }
    int rows[64];
    float coef[64];
    for (int i = 0; i < m->n_trees; i++) rows[i] = i;
    for (int t = 0; t < T; t++) {  // coefficient gelu(logit) + context; sum the node rows
        gelu_array(LB + (size_t)t * m->n_trees, coef, m->n_trees);
        for (int i = 0; i < m->n_trees; i++) coef[i] += CTX[(size_t)t * m->n_trees + i];
        tm_out_rows(&m->t_out, rows, coef, m->n_trees, OUT + (size_t)t * D);
    }
}

static void deep(const Mix *m, int T) {
    __m512i xq[16];
    int rows[256];
    float lg[256], coef[256];
    conv_quant(m, T);
    for (int t = 0; t < T; t++) {
        load_q(t, xq);
        int S = 0;
        for (int tr = 0; tr < m->n_trees; tr++) {
            int cur = 0;
            for (int k = 0; k < m->K; k++) {
                const int gid = tr * m->per_tree + cur;
                const float l = STEP[t] * tm_dot_i8(&m->t_in, gid, xq) + m->bias[gid];
                rows[S] = gid; lg[S++] = l;
                cur = 2 * cur + 1 + (l > 0);
            }
        }
        gelu_array(lg, coef, S);
        tm_out_rows(&m->t_out, rows, coef, S, OUT + (size_t)t * D);
    }
}

static double t_layers;

static void encode(const int *ids, int T, float *hidden) {
    for (int t = 0; t < T; t++) memcpy(X + (size_t)t * D, emb + (size_t)ids[t] * D, D * 4);
    const double t0 = now();
    for (int l = 0; l < NL; l++) {
        for (int t = 0; t < T; t++) rmsnorm(X + (size_t)t * D, layers[l].norm, U + (size_t)t * D, D);
        memset(OUT, 0, (size_t)T * D * 4);
        bank(&layers[l].bank, T);
        deep(&layers[l].deep, T);
        for (size_t i = 0; i < (size_t)T * D; i++) X[i] += OUT[i];
    }
    t_layers += now() - t0;
    for (int t = 0; t < T; t++) rmsnorm(X + (size_t)t * D, norm_f, hidden + (size_t)t * D, D);
}

static void head(const float *h, float *logits) {
    for (int v = 0; v < V; v++) logits[v] = dot(emb + (size_t)v * D, h, D) + head_bias[v];
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: enc_bench enc.bin [repeats]\n"); return 1; }
    const int reps = argc > 2 ? atoi(argv[2]) : 5;
    g_f = fopen(argv[1], "rb"); if (!g_f) { perror("open"); return 1; }
    if (rdi() != 7) { fprintf(stderr, "not an encoder file\n"); return 1; }
    V = rdi(); D = rdi(); NL = rdi(); TB = rdi(); HB = rdi(); NB = rdi(); TD = rdi(); DD = rdi(); KC = rdi();
    const int bits = rdi(); rdi();
    if (bits != 8 || NB != 16 || D % 64 || D > 1024) { fprintf(stderr, "unsupported shape\n"); return 1; }
    emb = rdf((size_t)V * D); head_bias = rdf(V); norm_f = rdf(D);
    layers = (Layer *)xalloc(NL * sizeof(Layer));
    for (int l = 0; l < NL; l++) {
        layers[l].norm = rdf(D);
        load_mix(&layers[l].bank, TB, 0, NB, HB, 1);
        load_mix(&layers[l].deep, TD, DD, 0, 1, 0);
    }
    const int T = rdi(), M = rdi();
    int *ids = (int *)xalloc(T * 4), *pos = (int *)xalloc(M * 4);
    if (fread(ids, 4, T, g_f) != (size_t)T || fread(pos, 4, M, g_f) != (size_t)M) return 1;
    float *ref = rdf((size_t)M * V);
    fclose(g_f);
    const int TMAX = 512 > T ? 512 : T;
    alloc_scratch(TMAX);
    float *hidden = (float *)xalloc((size_t)TMAX * D * 4), *logits = (float *)xalloc((size_t)V * 4);
    printf("GTS encoder: %d layers, width %d; bank %d trees (%d heads, state %d), %d deep trees of depth %d; %d tensors packed ternary\n",
           NL, D, TB, HB, NB, TD, DD, g_packed);

    // 1. same function as PyTorch at the masked positions
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
        nll_c += log(zc) + logits[a] - logits[b]; nll_r += log(zr);  // -log p(PyTorch's top-1) under each
    }
    printf("  check on %d tokens, %d masked: max |logit - PyTorch| = %.2e, same top-1 on %d/%d; "
           "-log p(PyTorch's top-1): kernel %.4f, PyTorch %.4f\n", T, M, worst, agree, M, nll_c / M, nll_r / M);

    // 2. timing: whole sequences of 128 and 512 tokens (the test tokens, repeated), and the head per masked token
    int *seq = (int *)xalloc(TMAX * 4);
    for (int t = 0; t < TMAX; t++) seq[t] = ids[t % T];
    const int lens[2] = {128, 512};
    for (int li = 0; li < 2; li++) {
        const int L = lens[li];
        encode(seq, L, hidden);  // warm
        double best = 1e30; t_layers = 0;
        for (int r = 0; r < reps; r++) { const double t0 = now(); encode(seq, L, hidden); const double dt = now() - t0; if (dt < best) best = dt; }
        printf("  sequence %3d: %.2f ms per sequence (best of %d) = %.1f us per token = %.0f tokens/s; layers %.1f us per token per layer\n",
               L, best * 1e3, reps, best / L * 1e6, L / best, t_layers / reps / L / NL * 1e6);
    }
    const double t0 = now();
    for (int r = 0; r < 20; r++) head(hidden + (size_t)(r % 128) * D, logits);
    printf("  masked-LM head (%d x %d, float32): %.2f ms per scored position\n", V, D, (now() - t0) / 20 * 1e3);
    return 0;
}
