// Checks that moth is causal, including its long global context (-DLCTX > TB): with the whole BLT forward built
// as moth's main builds it, change one input byte; every stage's values before it must stay bit-for-bit the same
// (encoder output, pooled patches, global stack output, decoder input, logits), and logits after it must move,
// including in later TB-byte chunks, which only the global stack reaches. Built with -DDIRECTCONV the long conv
// is a direct causal sum: with the FFT, rounding (~1e-7) reaches every position and now and then flips an int8
// rounding, a step with no information in it, which would blur an exact test. The integer engine, which
// generates, is causal by construction; moth checks it against the training forward at the end of every run.
//   cc -O2 -march=native -fopenmp -DDIRECTCONV -DLCTX=512 -DB=2 tests/moth_causal_check.c -o check -lm && ./check
#define main moth_main
#include "../moth.c"
#undef main

static float *cp(const float *x, size_t n) { float *y = fa(n); memcpy(y, x, n * 4); return y; }
int main(void) {
    NT = omp_get_max_threads(); if (NT > 64) NT = 64;
    int NMAX = B * LCTX;
    SCR = fa((size_t)NT * 6 * 2 * (TB > TG ? TB : TG) * CH); VB = fa((size_t)NMAX * D); DP = fa((size_t)3 * NMAX * D);
    HN = fa((size_t)NMAX * D); RF = fa(NMAX); LOGIT = fa((size_t)NMAX * V); S1 = fa((size_t)NMAX * D);
    G1 = fa((size_t)NMAX * D); WD = fa((size_t)V * D); WT = fa((size_t)V * D); G2 = fa((size_t)B * TG * D);
    arg = calloc((size_t)B * TG * D, 2); cpb = calloc(NMAX, sizeof(int));
    for (int k = 0; k < NG; k++) hidx[k] = calloc(NMAX, sizeof(int));
    Eh = param(V * D, 0.02f); stack_build(&SH, TB, LH, 1000, NW);
    Eb = param(V * D, 0.02f); for (int k = 0; k < NG; k++) Hs[k] = param(HV * D, 0.02f); Wo = param(V * D, 0.02f);
    stack_build(&SE, TB, LE, 2000, NW); stack_build(&SG, TG, LG, 3000, B); stack_build(&SD, TB, LD, 4000, NW);

    const char *txt = "The river rose in the night, and by morning the old stone bridge was gone. Villagers gathered on the bank "
                      "to argue about who would build the new one, and how, and with whose money. ";
    ndata = LCTX + 64; data = malloc(ndata); ENT = fa(ndata + 1);
    for (long i = 0; i < ndata; i++) { data[i] = txt[i % strlen(txt)]; ENT[i] = 4 * urand(); }   // fixed boundaries
    theta = 2.5f;
    long off[1] = {8}; const uint8_t *win[1] = {data + off[0]};
    float *base = fa((size_t)LCTX * V); blt_fwd(win, off, 1, 0); memcpy(base, LOGIT, (size_t)LCTX * V * 4);
    float *e0 = cp(SE.X[LE], (size_t)LCTX * D), *g0 = cp(SG.X[0], (size_t)TG * D), *z0 = cp(SG.X[LG], (size_t)TG * D), *d0 = cp(SD.X[0], (size_t)LCTX * D);
    printf("LCTX %d, TB %d (%d chunks), %d patches in the sequence\n", LCTX, TB, NCK, npat[0]);
    int bad = 0, at[] = {5, TB / 2, LCTX / 2 + 17, LCTX - TB / 2 - 3};
    for (int a = 0; a < 4; a++) {
        int k = at[a], pk = -1; for (int t = 0; t <= k; t++) pk += sb[0][t];   // the patch holding byte k
        data[off[0] + k] ^= 0x20; blt_fwd(win, off, 1, 0);
        double me = 0, mg = 0, mz = 0, md = 0, ml = 0, later = 0;
        for (size_t j = 0; j < (size_t)k * D; j++) { me = fmax(me, fabs(SE.X[LE][j] - e0[j])); md = fmax(md, fabs(SD.X[0][j] - d0[j])); }
        for (size_t j = 0; j < (size_t)k * V; j++) ml = fmax(ml, fabs(LOGIT[j] - base[j]));
        for (size_t j = 0; j < (size_t)pk * D; j++) { mg = fmax(mg, fabs(SG.X[0][j] - g0[j])); mz = fmax(mz, fabs(SG.X[LG][j] - z0[j])); }
        for (int t = (k / TB + 1) * TB; t < LCTX; t++) for (int c = 0; c < V; c++) later = fmax(later, fabs(LOGIT[(size_t)t * V + c] - base[(size_t)t * V + c]));
        int ok = me == 0 && mg == 0 && mz == 0 && md == 0 && ml == 0 && (k / TB == NCK - 1 || later > 1e-3);
        printf("byte %4d (chunk %d, patch %3d): before it, encoder %.0e pooled %.0e global %.0e decoder input %.0e logits %.0e; "
               "later chunks' logits move %.1e  %s\n", k, k / TB, pk, me, mg, mz, md, ml, later, ok ? "ok" : "FAIL");
        bad |= !ok; data[off[0] + k] ^= 0x20;
    }
    printf(bad ? "FAILED\n" : "all checks passed\n");
    return bad;
}
