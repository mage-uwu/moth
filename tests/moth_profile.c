// Where does moth's streaming inference spend its time? Loads a checkpoint, streams text one byte at a time
// through the real engine (VNNI kernels where built with them) on one thread, and times each stack and head;
// then times the Monarch products alone, as many as a byte needs, to split each stack into Monarchs and the rest
// (long and short convs, int8 quantisation, norms, GELU, residuals).
//   cc -O3 -march=native -mprefer-vector-width=512 -fopenmp -DHV=2048 tests/moth_profile.c -o prof -lm
//   OMP_NUM_THREADS=1 ./prof run.ck text.txt [bytes]
#define main moth_main
#include "../moth.c"
#undef main

static double ts[4], temb, thead[2];
static void timed_step(int byte, float *logits) {    // blt_step, with timers
    float e[D], h[D], d[D]; double t0 = now();
    if (NCK > 1 && bs.pos && bs.pos % TB == 0) { stack_reset(&SE); stack_reset(&SD); }
    memmove(bs.hist, bs.hist + 1, 7); bs.hist[7] = byte;
    memcpy(e, Eb->w + byte * D, sizeof e);
    for (int k = 0; k < NG; k++) { const float *r = Hs[k]->w + (size_t)hrow(bs.hist + 7, k + 3, k) * D; for (int i = 0; i < D; i++) e[i] += r[i]; }
    double t1 = now(); temb += t1 - t0;
    stack_step(&SE, e); double t2 = now(); ts[1] += t2 - t1;
    if (bs.start) { memcpy(bs.pm, e, sizeof e); bs.plen = 1; } else { for (int i = 0; i < D; i++) bs.pm[i] = MAXF(bs.pm[i], e[i]); bs.plen++; }
    memcpy(h, Eh->w + byte * D, sizeof h); t1 = now(); stack_step(&SH, h); t2 = now(); ts[0] += t2 - t1;
    head_step(h, &ih_ent, logits); t1 = now(); thead[0] += t1 - t2;
    int st = bs.pos == 0 || entropy_of(logits) > theta || bs.plen >= PMAX;
    t1 = now(); if (st) { memcpy(h, bs.pm, sizeof h); stack_step(&SG, h); memcpy(bs.z, h, sizeof h); bs.npatch++; } t2 = now(); ts[2] += t2 - t1;
    bs.start = st;
    for (int i = 0; i < D; i++) d[i] = e[i] + bs.z[i];
    t1 = now(); stack_step(&SD, d); t2 = now(); ts[3] += t2 - t1;
    head_step(d, &ih_out, logits); thead[1] += now() - t2;
    bs.pos++;
}
int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: prof run.ck text.txt [bytes]\n"); return 1; }
    long n = argc > 3 ? atol(argv[3]) : 200000;
    FILE *f = fopen(argv[2], "rb"); if (!f) return 1; fseek(f, 0, SEEK_END); long nf = ftell(f);
    if (n > nf - 16) n = nf - 16;
    uint8_t *txt = malloc(n + 8); fseek(f, nf - n - 8, SEEK_SET); if (fread(txt, 1, n + 8, f) != (size_t)n + 8) return 1; fclose(f);
    NT = 1; omp_set_num_threads(1);
    Eh = param(V * D, 0.02f); stack_build(&SH, TB, LH, 1000, NW);
    Eb = param(V * D, 0.02f); for (int k = 0; k < NG; k++) Hs[k] = param(HV * D, 0.02f); Wo = param(V * D, 0.02f);
    stack_build(&SE, TB, LE, 2000, NW); stack_build(&SG, TG, LG, 3000, B); stack_build(&SD, TB, LD, 4000, NW);
    SCR = fa((size_t)6 * 2 * (TB > TG ? TB : TG) * CH); WD = fa((size_t)V * D);
    ck_load(argv[1]);
    stack_prep(&SH); stack_prep(&SE); stack_prep(&SG); stack_prep(&SD); ihead_prep(&ih_ent, Eh->w); ihead_prep(&ih_out, Wo->w);
    float lg[V];
    blt_reset(txt); for (long t = 0; t < 2000; t++) timed_step(txt[8 + t], lg);      // warm up
    memset(ts, 0, sizeof ts); temb = thead[0] = thead[1] = 0;
    blt_reset(txt); double t0 = now();
    for (long t = 0; t < n; t++) timed_step(txt[8 + t], lg);
    double tt = now() - t0, per = tt / n * 1e9;
    // the Monarch products alone: 7 per layer, as many layers as a byte runs (the global stack's per patch)
    double lpb = LH + LE + LD + (double)LG * bs.npatch / n, mons = 7 * lpb;
    Layer *y = &SD.ly[0]; int8_t q[D]; float o[D]; long reps = 200000; volatile float sink = 0;
    for (int i = 0; i < D; i++) q[i] = (int8_t)((i * 37) % 255 - 127);
    Mon *ms[] = {&y->q, &y->k, &y->v, &y->o, &y->g, &y->u, &y->d};
    t0 = now(); for (long r = 0; r < reps; r++) { q[r % D] ^= 1; IMON(y, r % 7, ms[r % 7], q, 0.01f, o); sink += o[0]; }
    double tmon = (now() - t0) / reps * 1e9;
    printf("%ld bytes, %.2f bytes per patch: %.0f ns a byte = %.0f bytes/s on one thread (%s)\n", n, (double)n / bs.npatch, per, 1e9 / per,
#ifdef VNNI
           "VNNI kernels"
#else
           "portable C"
#endif
    );
    const char *nm[] = {"entropy model", "encoder", "global (per patch)", "decoder"};
    int nl[] = {LH, LE, LG, LD};
    printf("  embeddings          %5.1f%%\n", 100 * temb / tt);
    for (int s = 0; s < 4; s++) {
        double lay = s == 2 ? (double)LG * bs.npatch / n : nl[s], m = 7 * lay * tmon * n / 1e9;
        printf("  %-19s %5.1f%%  (Monarchs %4.1f%%, rest %4.1f%%)\n", nm[s], 100 * ts[s] / tt, 100 * m / tt, 100 * (ts[s] - m) / tt);
    }
    printf("  entropy head        %5.1f%%\n  output head         %5.1f%%\n", 100 * thead[0] / tt, 100 * thead[1] / tt);
    printf("  other (patching)    %5.1f%%\n", 100 * (tt - temb - ts[0] - ts[1] - ts[2] - ts[3] - thead[0] - thead[1]) / tt);
    printf("one Monarch product: %.0f ns; %.1f a byte = %.1f%% of the time%s\n", tmon, mons, 100 * mons * tmon / per, sink == 1234.5f ? "!" : "");
    return 0;
}
