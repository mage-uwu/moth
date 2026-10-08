# GTS-OMEGA: compiling the teacher's MLPs into tree-structured computation

Question: does ModernBERT-large's GeGLU MLP contain structure (a shared low-order map plus small, low-rank,
conditionally applied polynomial corrections) that a tree can exploit to keep more of the teacher's behaviour per unit
of compute than GTS's current trees + rank-128 affine path (Stage 2 tree error 0.277 -> 0.191)?

**Answer from the measurements below: not in the form proposed.** The algebra is exact and is verified, but the
teacher does not have the structure that would make it cheap. Its gates sit in GELU's smooth zone rather than in sign
regimes, its gate patterns are high-dimensional, its exact quadratic part neither stands alone nor compresses, its
Taylor series diverges on its own data, and its local Jacobian has rank ~300, beyond any 168-rank route. The family
that held up best end to end at fixed compute was the teacher's own top neurons kept in GeGLU form. The decisive test
left is a GPU A/B at equal training budget (section 6).

Code: `mamba_ssm/omega/reference.py` (conversions), `tests/omega/test_reference.py` (8 checks),
`scripts/omega_diagnostics.py` and `scripts/omega_polynomial.py` (measurements), results in
`results/omega/{diagnostics,polynomial}.json`. Teacher: `answerdotai/ModernBERT-large`, MLP input x = mlp_norm(h),
d = 1024, F = 2624, f(x) = O[(Ux) * h(Gx)], G = Wi[:F] (the GELU'd half), U = Wi[F:], O = Wo, h(t) = t Phi(t) (exact
erf GELU), no biases. Data: FineWeb-Edu, 32,768 tokens to fit, 8,192 held out, 4,096 masked tokens for the splice
test; layers 4, 10, 14 (a massive-activation layer), 17, 24. Error = relative squared error ||y_hat - y||^2 / ||y||^2,
Stage 2's metric.

## 1. Architecture and conversion algorithm

The proposal, made precise, with costs per token at d = 1024.

**Predicates.** pi_k(x) = [g_k . x > tau_k] over the teacher's own gate rows g_k (exact teacher switching surfaces);
thresholds tau_k from the hinge knots. Evaluated in a decision DAG: one variable per level, isomorphic subgraphs
merged (an ordered BDD over the chosen predicates), so repeated subcomputations are shared. Cost: one d-dot per
distinct predicate on the path.

**Shared component.** S(x) = affine of rank r_a + a CP-rank-r_q quadratic P3[(P1 x) * (P2 x)], fitted by least squares
on teacher activations (initialised from the teacher's highest-energy terms o_j a_j b_j). Cost d(2 r_a + 3 r_q).

**Node payloads (boundary-exact).** A node splitting on pi_k carries
Delta_v(x) = (g_k . x - tau_k) W_v P_v x, with W_v P_v of rank k_v: zero on the node's own hyperplane, so neighbouring
regions agree there by construction. This is the hinge expansion's switched term c_k o_j a_j (b_j - tau_k)[b_j > tau_k]
with a_j generalised to a learned rank-k_v projection. Output: S(x) + sum over the path of Delta_v(x) (the telescoping
identity, section 2). Cost per node d(1 + 2 k_v) (the b_k is shared with routing).

**Conversion.** (1) hinge-compile the teacher (exact representation of f_Delta, section 2); (2) collect teacher
activations; (3) fit S; (4) grow the DAG greedily over teacher-gate predicates, choosing at each step the predicate and
rank that most reduce held-out residual energy per added multiply-add; (5) fit each Delta_v by least squares within the
boundary-exact family; (6) merge nodes whose payloads agree to tolerance; (7) quantise (ternary / int8); (8) Stage 2
fine-tune against the teacher sub-block, then Stage 3 end to end.

**Storage.** S: d(2 r_a + 3 r_q); per DAG node: d(2 k_v) plus a predicate index and threshold.

**Local rank budget.** d/dx[(g . x - tau) W P x] = W P x g^T + (g . x - tau) W P has rank <= k + 1; the quadratic core's
Jacobian P3 diag(P2 x) P1 + P3 diag(P1 x) P2 has rank <= r_q. So rank J <= r_a + r_q + D(k + 1) along a depth-D path.

## 2. Exactness ledger

| step | status | preserves | cost relative to the teacher |
|---|---|---|---|
| f(x) = (f(x) + f(-x))/2 + (f(x) - f(-x))/2, even part = Q/2, Q = sum_j o_j a_j b_j | exact (h(t) - h(-t) = t) | everything | representation only |
| odd part = 1/2 sum_j o_j a_j b_j erf(b_j / sqrt 2) | exact | everything | same terms as the teacher |
| f = E_Z sum_j o_j a_j b_j 1[b_j >= Z], one shared Z ~ N(0,1) | exact | everything | evaluating the expectation is Phi again: no saving |
| GELU -> hinge h_Delta on symmetric knots in [-T, T], tails of slope 0 / 1 | controlled: sup error <= max(sqrt(2/pi) Delta^2 / 8, T Phi(-T)) (T = 4, Delta = 0.25: 6.2e-3 / 1.3e-4); ||f - f_Delta|| <= eps ||O||_op ||Ux|| | parity h_Delta(t) - h_Delta(-t) = t, hence Q/2 exactly; continuity | F (K + 1) switched terms: more than the teacher |
| f_Delta = sum of threshold-gated quadratic pieces c_k o_j a_j (b_j - tau_k)[b_j > tau_k] | exact given the hinge | continuity at every threshold | none |
| ReGLU teacher: f = Q_s(x) per sign region, neighbouring regions agree on the hyperplane | exact for relu only; for this GELU teacher an approximation (error 0.002-0.30, section 3) | continuity | none |
| Q_root + sum_path (Q_v - Q_parent) = Q_leaf | exact | everything | saving only if the corrections are small or low-rank (empirical; section 3: they are not) |
| shared core S at CP rank r_q, routed corrections of rank k | approximation (structural compression) | boundary continuity (by construction) | the only step that saves compute |
| ternary / int8 weights | approximation (quantisation) | - | storage and speed |
| GTS route (affine rank 128 + 40 nodes) | exact statement: rank J <= 168 within a route | - | rules out local equivalence where the teacher's rank exceeds 168 (section 3: ~300); not an output-error floor |

Representational equivalence and computational savings are separate columns on purpose: every exact step above costs
at least as much as the teacher; savings come only from the approximate compression steps, whose value is empirical.

## 3. Geometry diagnostics on teacher activations

| | layer 4 | layer 10 | layer 14 | layer 17 | layer 24 |
|---|---|---|---|---|---|
| **Input distribution** | | | | | |
| PCA rank for 90% / 99% variance | 632 / 953 | 533 / 928 | 579 / 942 | 594 / 948 | 551 / 935 |
| kurtosis, random projections (Gaussian 3) | 3.22 | 3.14 | 3.10 | 3.11 | 3.07 |
| **Parity** | | | | | |
| even part Q/2 alone (rel. sq. error) | 1.51 | 1.29 | 0.28 | 3.21 | 0.67 |
| odd part alone | 2.77 | 1.83 | 0.28 | 3.60 | 1.00 |
| cos(even, odd) on real x | -0.80 | -0.69 | +0.78 | -0.86 | -0.41 |
| **Affine** | | | | | |
| least squares on real x | 0.267 | 0.281 | 0.176 | 0.381 | 0.262 |
| Gaussian-Hermite affine (matched mean, cov), on real x | 0.369 | 0.668 | 0.994 | 0.784 | 0.465 |
| **Gates b = Gx (2,624)** | | | | | |
| per token with \|b\| < 1 / < 0.25 | 2493 / 1004 | 2297 / 835 | 2094 / 669 | 2041 / 646 | 2029 / 662 |
| always on / always off (>98% / <2% of tokens) | 98 / 110 | 197 / 97 | 275 / 81 | 260 / 54 | 15 / 10 |
| output energy from switching gates | 0.75 | 0.73 | 0.05 | 0.73 | 0.66 |
| sign-pattern rank for 90% (participation ratio) | 1461 (98) | 1465 (168) | 1464 (159) | 1450 (119) | 1344 (130) |
| gate values outside [-4, 4] | 2.5e-05 | 5.4e-06 | 7.8e-04 | 4.0e-04 | 1.5e-03 |
| **Surrogates of h** | | | | | |
| ReGLU (relu for gelu): sign-region floor | 0.303 | 0.201 | 0.002 | 0.178 | 0.032 |
| hinge GELU T=4, Delta 0.5 / 0.25 | 3.8e-03 / 2.4e-04 | 3.6e-03 / 2.3e-04 | 2.3e-05 / 1.5e-06 | 2.3e-03 / 1.5e-04 | 2.5e-04 / 1.6e-05 |
| GELU series to degree 3 / 5 in x | 1.2 / 5.4 | 0.92 / 2.1 | 1.2e+02 / 4.4e+06 | 7.3 / 77 | 1.6 / 55 |
| **Quadratic Q (error on Q)** | | | | | |
| teacher's top 64 / 256 / 1024 terms | 0.271 / 0.173 / 0.065 | 0.360 / 0.177 / 0.054 | 0.027 / 0.013 / 0.004 | 0.171 / 0.089 / 0.033 | 0.240 / 0.145 / 0.061 |
| fitted CP rank 64 / 256 | 0.234 / 0.137 | 0.289 / 0.136 | 0.021 / 0.010 | 0.135 / 0.071 | 0.212 / 0.126 |
| Q's output rank for 90% | 310 | 275 | 1 | 245 | 320 |
| **Jacobian (24 held-out tokens, medians)** | | | | | |
| rank for 90% of energy: J / J - affine / J on input PCs | 319 / 293 / 270 | 318 / 307 / 231 | 309 / 1 / 235 | 319 / 329 / 248 | 283 / 354 / 227 |
| best rank-168 relative Frobenius error (GTS route cap) | 0.51 | 0.51 | 0.50 | 0.52 | 0.48 |
| **Sign-region quadratics** | | | | | |
| error at tree depth 0 / 3 / 8 | 0.498 / 0.476 / 0.417 | 0.508 / 0.338 / 0.318 | 0.851 / 0.008 / 0.004 | 0.567 / 0.463 / 0.394 | 0.328 / 0.289 / 0.272 |
| child - parent energy (median) / output rank90 | 0.026 / 15 | 0.057 / 18 | 0.011 / 13 | 0.077 / 12 | 0.010 / 10 |

What the rows say:

- **The quadratic even part is exact but not a usable component.** On real x the even and odd parts are each larger
  than f and cancel (cos -0.41 to -0.86; layer 14 the exception). Most gates are off (b < 0), where h(b) ~ 0 and the
  term's halves +-ab/2 cancel. Q itself needs ~1,000 teacher terms or CP rank > 256 for ~0.06-0.14 error on Q.
- **The routing algebra is high-dimensional.** 2,000-2,500 of 2,624 gates sit in |b| < 1 on every token, the sign
  pattern needs ~1,340-1,470 dimensions for 90% of its variance, and only 25-356 gates are effectively constant. The
  sign-region family's floor (ReGLU) is 0.18-0.30 error except at layers 14 and 24.
- **Regional corrections are not small enough to pay.** Sign-region quadratics along a greedy predicate tree improve
  from depth 0 to 8 by 0.06-0.19 (layer 14: 0.85 -> 0.004, its massive direction comes from a few switching neurons),
  with child-minus-parent corrections of 1-8% energy and output rank ~10-18 each: many corrections, none dominant.
- **No low-degree global polynomial either.** GELU's Taylor series in b diverges on the 5-20% of gate values beyond
  |b| > 1 (times large a_j): degree 3 gives 0.9-120, degree 5 worse. Saturation is essential.
- **The affine effect is a property of the data, not of Gaussian statistics.** Inputs look Gaussian in their
  marginals (kurtosis ~3.1), but the Hermite (matched-Gaussian) affine map scores 0.37-0.99 on real x against
  0.18-0.38 for least squares on real x.
- **Local rank ~300.** The teacher Jacobian needs rank ~283-319 for 90% of its energy (227-270 on the input's own
  principal subspace); the best rank-168 approximation leaves ~0.5 relative Frobenius error at every layer. Trees +
  affine cannot be locally equivalent; a rank-168 route is far from the teacher's local map.
- **The hinge conversion is the one near-lossless step**: Delta = 0.25 gives 1.5e-6 to 2.4e-4; gate values outside
  [-4, 4] are 5e-6 to 1.5e-3 and covered by the tail slopes (bound above).

## 4. Reference prototype

`mamba_ssm/omega/reference.py`, verified in float64 by `tests/omega/test_reference.py` on random GeGLU MLPs:
even/odd parts (1e-12), shared-Z expectation (Monte Carlo, 1%), hinge sup error under its bound over [-12, 12] with
exact parity h_Delta(t) - h_Delta(-t) = t and interpolation at the knots, ||f - f_Delta|| under eps ||O|| ||Ux|| row by
row with the even part preserved exactly, the switched-term sum equal to f_Delta and continuous across a switching
hyperplane, ReGLU region maps and their agreement on a one-gate boundary, telescoping, and the analytic Jacobian
against autograd.

## 5. Controlled evaluation (teacher attention kept, one MLP replaced)

Families fitted on the same 32,768 tokens (Adam, 600 steps, initialised from teacher structure), scored on held-out
tokens (local error) and spliced into the teacher at that layer alone (KL(teacher || spliced) at 15% masked positions
of 4,096 held-out tokens; teacher CE 1.074). Entries: **local error / splice KL**.

| family (multiply-adds / token) | layer 4 | layer 10 | layer 14 | layer 17 | layer 24 |
|---|---|---|---|---|---|
| affine rank 128 (262K) | 0.349 / **0.113** | 0.374 / **0.043** | 0.177 / **1.541** | 0.510 / **0.032** | 0.316 / **0.129** |
| dense GeGLU 110 (teacher's top neurons, refit) (338K) | 0.342 / **0.032** | 0.363 / **0.033** | 0.005 / **0.027** | 0.518 / **0.028** | 0.297 / **0.130** |
| affine 128 + CP quadratic 64 (459K) | 0.305 / **0.135** | 0.323 / **0.039** | 0.012 / **0.329** | 0.463 / **0.031** | 0.281 / **0.119** |
| OMEGA-lite: + 6 gate predicates, rank-4 bilinear leaves (477K) | 0.301 / **0.138** | 0.320 / **0.039** | 0.011 / **0.338** | 0.458 / **0.031** | 0.275 / **0.120** |
| dense GeGLU 256 (786K) | 0.256 / **0.029** | 0.267 / **0.025** | 0.004 / **0.023** | 0.396 / **0.024** | 0.230 / **0.114** |
| GTS trees + affine (GPU A/B, 47M tokens; ~344K) | 0.256 / n.a. | 0.246 / n.a. | 0.013 / n.a. | 0.364 / n.a. | 0.150 / n.a. |
| GTS trees alone (GPU A/B; ~82K) | 0.394 / n.a. | 0.373 / n.a. | 0.012 / n.a. | 0.519 / n.a. | 0.201 / n.a. |
| teacher (8,061K) | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

- At ~340K-480K multiply-adds the four fitted families are within a few points of each other locally, but end to end
  the teacher-neuron GeGLU is clearly best where it matters (layer 4: KL 0.032 vs 0.113-0.138; layer 14: 0.027 vs
  0.33 for the polynomial families and 1.54 for affine) and tied elsewhere. Lower local error from polynomial terms
  did not buy lower KL: unbounded terms extrapolate (masked tokens, other layers' errors) where GELU saturates.
- OMEGA-lite (shared quadratic + routed boundary-free bilinear leaves) adds almost nothing over its shared core.
- The GPU-trained trees + affine (0.150-0.364) beat every CPU-fitted family locally, but with ~1,400x the training
  tokens; their splice KL was not measured (checkpoints not kept). Equal-budget training is the open comparison.
- Latency and memory (C runtime, AMX, 4 vCPUs, results/gtsl_cpu_bench.log): trees + affine ~2 us per token per layer;
  a dense ternary/int8 GeGLU layer of the teacher's full width costs ~2-4 us more; a 110-256-neuron GeGLU is 4-10% of
  the full width, so a few tenths of a microsecond on the same GEMM (an estimate from the measured GEMM rate, not timed).
  Storage per layer: affine 128 262K weights; GeGLU 110 / 256: 338K / 786K; trees 8.4M (sparse, 1% read per token).

## 6. Falsifiable recommendation

**Do not invest further in region-polynomial OMEGA for this teacher.** It would have been justified by: a Boolean regime
(few gates near zero), a low-dimensional sign pattern, regional corrections that are low-rank and concentrated, or a
compressible Q. All four measured negative at four of five layers.

**What the evidence favours instead is the simpler architecture:** a narrow dense GeGLU initialised from the teacher's
own highest-energy neurons (MOHAWK's weight transfer, applied to the MLP), alone or beside the trees.

**The decisive experiment** (one A100, ~$2): Stage 2 on GPU at EVA's settings, 47M tokens per arm, four arms at
~340K multiply-adds per token: (a) trees + rank-128 affine (current), (b) dense GeGLU of the teacher's top 110 neurons,
(c) GeGLU 64 + trees, (d) the same with GeGLU 256 for the cost curve; per-layer held-out error, then a splice test of
each arm's layers into the teacher and a short Stage 3. Decision rule:
- if (b) or (c) beats (a) by >= 15% in Stage 2 error **and** in splice KL, replace or augment the trees with teacher-
  initialised GeGLU (it runs on the AMX GEMM, so the CPU cost is the same order as trees + affine);
- if (a) wins at equal budget, the trees earn their place and GTS's tree thesis stands for this teacher;
- in either case region-polynomial routing (OMEGA as proposed) stays parked unless a future teacher shows a Boolean
  regime (ReGLU floor < 0.05 and sign-pattern rank < ~200), which layer 14 alone does here.
