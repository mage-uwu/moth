// Checks bmoth.c's sequence mixer, bidirectional and causal, against direct sums: the FFT long conv (both
// halves) and the short convs forward, and their backward passes by the adjoint identity, which holds exactly
// for linear maps: <G, conv(u)> = <du, u> = <dh, h> + <dh_b, h_b>, and <g, shortconv(p)> = <dp, p>.
// Runs with and without padded sequences (the global stack's case).
//   cc -O2 -march=native -fopenmp tests/bmoth_conv_check.c -o check -lm && ./check
#define main bmoth_main
#include "../bmoth.c"
#undef main

int flow_check(void);
static double rnd(void) { return urand() * 2 - 1; }
static int check(int bi, int padded) {
    np = 0;                                     // each check builds afresh in the fixed parameter table
    Stack S; stack_build(&S, TB, 1, 77 + bi, bi); Layer *y = &S.ly[0]; Conv *c = &y->cv;
    int T = TB, N = S.N, nb = B, n = nb * T, lens[B], bad = 0;
    for (int b = 0; b < B; b++) lens[b] = padded ? 1 + (b * 37) % T : T;
    S.len = padded ? lens : NULL;
    for (int w = 0; w < 3; w++) for (int t = 0; t < n; t++) for (int ch = 0; ch < D; ch++)
        y->pre[((size_t)w * N + t) * D + ch] = t % T < lens[t / T] ? rnd() : 0;   // stack_fwd zeroes padding
    c->amax[0] = c->amax[1] = c->amax[2] = 1; c->amax[3] = 0.5f;
    mixer_fwd(&S, y, nb, 0);
    float *u = fa((size_t)n * D), *cc = fa((size_t)n * D), *pq = fa((size_t)3 * N * D), *post = fa((size_t)3 * N * D);
    memcpy(u, VB, (size_t)n * D * 4); memcpy(cc, y->cc, (size_t)n * D * 4);
    memcpy(pq, y->pre, (size_t)3 * N * D * 4); memcpy(post, y->post, (size_t)3 * N * D * 4);
    double e1 = 0, m1 = 0, e2 = 0, m2 = 0;
    for (int b = 0; b < nb; b++) for (int t = 0; t < lens[b]; t++) for (int ch = 0; ch < D; ch++) {
        double d = 0; size_t o = (size_t)b * T * D + ch;             // long conv, directly
        for (int j = 0; j <= t; j++) d += c->he[j * D + ch] * u[o + (size_t)(t - j) * D];
        if (bi) for (int j = 0; t + j < T; j++) d += c->heb[j * D + ch] * u[o + (size_t)(t + j) * D];
        e1 = fmax(e1, fabs(d - cc[o + (size_t)t * D])); m1 = fmax(m1, fabs(d));
        for (int w = 0; w < 3; w++) {                                // short convs, directly
            double s = c->sbe[w * D + ch];
            for (int j = 0; j < 3; j++) { int st = t - j + bi; if (st >= 0 && st < T) s += c->swe[j * 3 * D + w * D + ch] * pq[((size_t)w * N + b * T + st) * D + ch]; }
            e2 = fmax(e2, fabs(s - post[((size_t)w * N + b * T + t) * D + ch])); m2 = fmax(m2, fabs(s));
        }
    }
    // backward: G = grad of the conv output (zero on padding, as stack_bwd leaves it), random dq
    float *G = fa((size_t)n * D);
    for (int t = 0; t < n; t++) for (int ch = 0; ch < D; ch++) G[(size_t)t * D + ch] = y->cc[(size_t)t * D + ch] = t % T < lens[t / T] ? rnd() : 0;
    for (int t = 0; t < n; t++) for (int ch = 0; ch < D; ch++) DP[(size_t)t * D + ch] = t % T < lens[t / T] ? rnd() : 0;
    for (int ch = 0; ch < D; ch++) c->be[ch] = 0;                   // leave the skip term out of du
    mixer_bwd(&S, y, nb);
    double gc = 0, du = 0, dh = 0, sp = 0, dp = 0;
    for (size_t i = 0; i < (size_t)n * D; i++) { gc += (double)G[i] * cc[i]; du += (double)VB[i] * u[i]; }
    for (int t = 0; t < T; t++) for (int ch = 0; ch < D; ch++) {    // filter_bwd scaled dh by mod in place
        size_t i = (size_t)t * D + ch; dh += c->dh[i] / c->mod[i] * c->he[i] + (bi ? c->dhb[i] / c->mod[i] * c->heb[i] : 0);
    }
    for (int w = 0; w < 3; w++) for (int t = 0; t < n; t++) for (int ch = 0; ch < D; ch++) {
        size_t i = ((size_t)w * N + t) * D + ch;
        if (t % T < lens[t / T]) sp += (double)DP[i] * (post[i] - c->sbe[w * D + ch]);
        dp += (double)y->pre[i] * pq[i];
    }
    double r1 = e1 / m1, r2 = e2 / m2, a1 = fabs(du - gc) / fabs(gc), a2 = fabs(dh - gc) / fabs(gc), a3 = fabs(dp - sp) / fabs(sp);
    bad = r1 > 1e-4 || r2 > 1e-5 || a1 > 1e-4 || a2 > 1e-4 || a3 > 1e-5;
    printf("%-13s %-8s long conv fwd %.1e | short conv fwd %.1e | adjoint du %.1e dh %.1e | short conv bwd %.1e  %s\n",
           bi ? "bidirectional" : "causal", padded ? "padded" : "full", r1, r2, a1, a2, a3, bad ? "FAIL" : "ok");
    return bad;
}
int main(void) {
    NT = omp_get_max_threads(); if (NT > 64) NT = 64;
    int NMAX = B * TB;
    SCR = fa((size_t)NT * SW * 2 * TB * CH); VB = fa((size_t)NMAX * D); DP = fa((size_t)3 * NMAX * D);
    int bad = 0;
    for (int bi = 0; bi < 2; bi++) for (int pad = 0; pad < 2; pad++) bad |= check(bi, pad);
    bad |= flow_check();
    printf(bad ? "FAILED\n" : "all checks passed\n");
    return bad;
}

// Whole-model information flow: with the full BLT built as bmoth's main builds it (fresh weights), change one
// input byte and see which output positions' logits move. Bidirectional: positions on both sides must move.
int flow_check(void) {
    np = 0; Eh = param(V * D, 0.02f); stack_build(&SH, TB, LH, 1000, 0);
    Eb = param(V * D, 0.02f); for (int k = 0; k < NG; k++) Hs[k] = param(HV * D, 0.02f); Wo = param(V * D, 0.02f);
    stack_build(&SE, TB, LE, 2000, 1); stack_build(&SG, TG, LG, 3000, 1); stack_build(&SD, TB, LD, 4000, 1);
    HN = fa((size_t)B * TB * D); RF = fa(B * TB); LOGIT = fa((size_t)B * TB * V); S1 = fa((size_t)B * TB * D);
    G1 = fa((size_t)B * TB * D); WD = fa((size_t)V * D); WT = fa((size_t)V * D); G2 = fa((size_t)B * TG * D);
    arg = calloc((size_t)B * TG * D, 2); cpb = calloc(B * TB, sizeof(int)); for (int k = 0; k < NG; k++) hidx[k] = calloc(B * TB, sizeof(int));
    theta = 5.5f;                                               // untrained entropy model: some finite patch length
    uint8_t buf[8 + TB]; const char *txt = "The quick brown fox jumps over the lazy dog, and the dog, being lazy, does not mind at all. Then the fox runs home to its den.";
    memset(buf, '\n', sizeof buf); memcpy(buf + 8, txt, strlen(txt) < TB ? strlen(txt) : TB);
    const uint8_t *w[1] = {buf + 8};
    float *base = fa((size_t)TB * V); blt_fwd(w, 1, 0); memcpy(base, LOGIT, (size_t)TB * V * 4);
    int at = 64; buf[8 + at] ^= 0x20;                           // flip the case of the byte at 64
    blt_fwd(w, 1, 0);
    double left = 0, right = 0;
    for (int t = 0; t < TB; t++) for (int c = 0; c < V; c++) {
        double d = fabs(LOGIT[(size_t)t * V + c] - base[(size_t)t * V + c]);
        if (t < at) left = fmax(left, d); else if (t > at) right = fmax(right, d);
    }
    printf("changing byte %d moves logits before it by up to %.2e, after it by up to %.2e  %s\n", at, left, right, left > 1e-4 && right > 1e-4 ? "ok" : "FAIL");
    return !(left > 1e-4 && right > 1e-4);
}
