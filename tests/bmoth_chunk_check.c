// Checks bmoth.c's long-context wiring (-DLCTX, -DOV): the byte stacks run on overlapping chunks and each byte is
// read from the chunk that owns it.
//  1. Same model two ways. Built with -DLCTX=128 -DOV=32, a sequence is two chunks that are both the whole
//     128-byte window (the second is clamped back to byte 0), each owning half the bytes: the forward must
//     reproduce the default build's, and the parameter gradients, now gathered from the two chunks' halves,
//     its gradients. -DSRNEAREST rounds gradient codes to nearest: stochastic rounding keys its noise on the row
//     index, which the chunks change. Run the default build with "write", then the chunked one with "check":
//       cc -O2 -march=native -fopenmp -DSRNEAREST tests/bmoth_chunk_check.c -o a -lm && ./a write ref.bin
//       cc -O2 -march=native -fopenmp -DSRNEAREST -DLCTX=128 -DOV=32 tests/bmoth_chunk_check.c -o b -lm && ./b check ref.bin
//  2. Reach, in a real multi-chunk build: changing a byte in the first chunk moves the logits of the last chunk
//     and the other way round (only the global stack connects them), and the model trains a few steps:
//       cc -O2 -march=native -fopenmp -DLCTX=1024 -DB=2 tests/bmoth_chunk_check.c -o c -lm && ./c reach
#define main bmoth_main
#include "../bmoth.c"
#undef main

static int ke0, ke1;                            // embedding and encoder parameters: [ke0, ke1)
static void setup(void) {
    NT = omp_get_max_threads(); if (NT > 64) NT = 64;
    int NMAX = NW * TB;
    SCR = fa((size_t)NT * SW * 2 * (TB > TG ? TB : TG) * CH); VB = fa((size_t)NMAX * D); DP = fa((size_t)3 * NMAX * D);
    HN = fa((size_t)NMAX * D); RF = fa(NMAX); LOGIT = fa((size_t)NMAX * V); S1 = fa((size_t)NMAX * D);
    G1 = fa((size_t)NMAX * D); WD = fa((size_t)V * D); WT = fa((size_t)V * D); G2 = fa((size_t)B * TG * D);
    arg = calloc((size_t)B * TG * D, 2); cpb = calloc(NMAX, sizeof(int)); HO = fa((size_t)B * LCTX * D); GH = fa((size_t)B * LCTX * D);
    chunk_maps(); for (int k = 0; k < NG; k++) hidx[k] = calloc(NMAX, sizeof(int));
    Eh = param(V * D, 0.02f); stack_build(&SH, TB, LH, 1000, 0, NW);
    ke0 = np; Eb = param(V * D, 0.02f); for (int k = 0; k < NG; k++) Hs[k] = param(HV * D, 0.02f); Wo = param(V * D, 0.02f);
    stack_build(&SE, TB, LE, 2000, 1, NW); ke1 = np; stack_build(&SG, TG, LG, 3000, 1, B); stack_build(&SD, TB, LD, 4000, 1, NW);
    const char *txt = "The river rose in the night, and by morning the old stone bridge was gone. Villagers gathered on the bank "
                      "to argue about who would build the new one, and how, and with whose money. \"Stone,\" said the smith. ";
    ndata = 64 * LCTX; data = malloc(ndata); ntrain = ndata * 9 / 10;
    for (long i = 0; i < ndata; i++) data[i] = txt[(i * 7 + i / 131) % strlen(txt)];
    theta = 1.0f;                               // an untrained entropy model: some finite patch length
}
static int same_model(int wr, const char *path) {
    const uint8_t *win[B], *tgt[B]; long off[B]; int bad = 0;
    float loss = 0;
    for (int step = 0; step < 2; step++) {      // step 0 sets the delayed gradient scales (training discards it too)
        for (int k = 0; k < np; k++) memset(ps[k].g, 0, ps[k].n * sizeof(float));
        seed = hash(99 + step); for (int b = 0; b < B; b++) { off[b] = 100 + 37 * b + 1000 * step; tgt[b] = data + off[b]; }
        mask_windows(off, B, win); blt_fwd(win, B, 1);
        loss = xent_masked(tgt, B * LCTX, 1); blt_bwd(win, B);
    }
    FILE *f = fopen(path, wr ? "wb" : "rb"); if (!f) { fprintf(stderr, "no %s\n", path); return 1; }
    if (wr) {
        fwrite(&loss, 4, 1, f); for (int k = 0; k < np; k++) fwrite(ps[k].g, 4, ps[k].n, f);
        fwrite(G2, 4, (size_t)B * TG * D, f);
        printf("default build: masked loss %.6f, %d parameter tensors' gradients written to %s\n", loss, np, path);
    } else {
        float l0; if (fread(&l0, 4, 1, f) != 1) return 1;
        double worst = 1, rest = 1; int wk = 0;
        for (int k = 0; k < np; k++) {                  // cosine between each tensor's two gradients
            float *g = malloc(ps[k].n * 4); if (fread(g, 4, ps[k].n, f) != (size_t)ps[k].n) return 1;
            double e = 0, m = 0, xy = 0; for (int i = 0; i < ps[k].n; i++) { e += (double)ps[k].g[i] * ps[k].g[i]; m += (double)g[i] * g[i]; xy += (double)g[i] * ps[k].g[i]; }
            double cs = m > 0 ? xy / sqrt(e * m + 1e-300) : 1;
            if (k >= ke0 && k < ke1 && cs < worst) { worst = cs; wk = k; }
            if ((k < ke0 || k >= ke1) && m > 0 && cs < rest) rest = cs;
            if (getenv("VERBOSE") && m > 0) printf("  param %3d (%7d values): cosine %.5f\n", k, ps[k].n, cs);
            free(g);
        }
        float *g2 = fa((size_t)B * TG * D); double xy = 0, xx = 0, yy = 0;   // grad wrt the pooled patches (the global stack's input)
        if (fread(g2, 4, (size_t)B * TG * D, f) != (size_t)B * TG * D) return 1;
        for (size_t i = 0; i < (size_t)B * TG * D; i++) { xy += (double)g2[i] * G2[i]; xx += (double)g2[i] * g2[i]; yy += (double)G2[i] * G2[i]; }
        double cg = xy / sqrt(xx * yy);
        // Gradients run through int8 codes, and each chunk quantises its share of a token's gradient on its own,
        // so they can't match exactly; how closely they can is set by rounding: switching the default build between
        // stochastic and nearest rounding alone puts the pooled-patch gradient at cosine ~0.6 and many parameter
        // tensors below 0.9 (untrained, many are near-cancelling sums). The routing this build adds sits between
        // the head and the pooled patches, and between them and the embeddings: those must agree closely
        bad = fabsf(loss - l0) > 1e-4f * l0 || cg < 0.99 || worst < 0.99;
        printf("chunked build (%d chunks of %d, owning %d bytes each): masked loss %.6f vs %.6f\n"
               "  gradient wrt the pooled patches: cosine %.5f, norm ratio %.4f\n"
               "  embedding and encoder parameter gradients: lowest cosine %.4f (param %d)\n"
               "  (global and decoder parameter gradients, not checked: lowest cosine %.3f)  %s\n",
               NCK, TB, CS, loss, l0, cg, sqrt(yy / xx), worst, wk, rest, bad ? "FAIL" : "ok");
    }
    fclose(f); return bad;
}
static int reach(void) {
    static uint8_t buf[8 + LCTX]; const uint8_t *w[1] = {buf + 8}; int bad = 0;
    memcpy(buf, data + 100, 8 + LCTX);
    float *base = fa((size_t)LCTX * V); blt_fwd(w, 1, 0); memcpy(base, LOGIT, (size_t)LCTX * V * 4);
    int at[2] = {5, LCTX - 6};                                           // in the first chunk, in the last
    for (int a = 0; a < 2; a++) {
        buf[8 + at[a]] ^= 0x20; blt_fwd(w, 1, 0); buf[8 + at[a]] ^= 0x20;
        int lo = a ? 0 : LCTX - CS, hi = a ? CS : LCTX; double mv = 0;   // the far chunk's owned bytes
        for (int t = lo; t < hi; t++) for (int c = 0; c < V; c++) mv = fmax(mv, fabs(LOGIT[(size_t)t * V + c] - base[(size_t)t * V + c]));
        printf("changing byte %d moves the logits of bytes %d-%d by up to %.2e  %s\n", at[a], lo, hi - 1, mv, mv > 1e-4 ? "ok" : "FAIL");
        bad |= !(mv > 1e-4);
    }
    const uint8_t *win[B], *tgt[B]; long off[B]; float l0 = 0, l1 = 0;   // a few training steps: loss must fall
    for (int step = -1; step <= 40; step++) {
        seed = hash(step + 7); for (int b = 0; b < B; b++) { off[b] = pick(0); tgt[b] = data + off[b]; }
        mask_windows(off, B, win); blt_fwd(win, B, 1); float l = xent_masked(tgt, B * LCTX, 1); blt_bwd(win, B); adam(step, 0, np, 40);
        if (step == 0) l0 = l;
        l1 = l;
    }
    printf("40 training steps: masked loss %.3f -> %.3f  %s\n", l0, l1, l1 < l0 ? "ok" : "FAIL");
    return bad | !(l1 < l0);
}
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: write|check ref.bin, or reach\n"); return 1; }
    setup();
    printf("LCTX %d, TB %d, OV %d: %d chunks a sequence, %d patch slots\n", LCTX, TB, OV, NCK, TG);
    int bad = !strcmp(argv[1], "reach") ? reach() : same_model(!strcmp(argv[1], "write"), argc > 2 ? argv[2] : "ref.bin");
    printf(bad ? "FAILED\n" : "all checks passed\n");
    return bad;
}
