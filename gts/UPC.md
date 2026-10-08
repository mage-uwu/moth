# GTS3-UPC-10M

A 9,967,276-parameter GTS3-style mixed forest adapted to typed VM effects. This extension imports the existing `mamba_ssm.modules.gts.GTSMixed`; it does not modify the mixer or the GTS3 checkpoint.

**Status: learned instruction decoding/binding pilot, not a universal planner.** The network receives an explicitly named opcode and symbolic operands, and predicts the opcode plus physical register pointers. The host parses syntax, advances the program counter, stores memory and performs conventional arithmetic. High binding accuracy does not demonstrate inventing a policy or interpreting underspecified goals.

## Architecture

GTS3 is the 112.9M phase-3 masked-LM checkpoint documented in `checkpoints/bert110m/README.md`. This adaptation transfers its architecture, not its trained weights.

| Component | UPC configuration |
|---|---|
| Width / layers | 256 / 9 |
| Context bank | 32 depth-zero trees, 8 heads, state 16 |
| Stateless forest | 4 depth-8 trees per block |
| Convolution | Centered depthwise width 3 |
| Quantization | Group-128 absmean ternary; q8 activations |
| Routing | Hard root-to-leaf forward; routing STE in training |
| Inputs | 256 symbols, 32 opcodes, typed roles, numeric slot features |
| Outputs | Opcode and four variable-length pointers |
| English tokenizer / game ID | None |

The stateless component visits 36 of 2,044 nodes per token per layer. This is not a claim that total inference cost is reduced by that ratio: the bank, projections, convolutions, heads and runtime have additional costs.

## Repository-native run

From `gts/`, with torch, numpy and pytest installed:

```sh
python -m pytest tests/modules/test_gts_upc.py -q
python scripts/upc_pretrain.py --steps 900 --batch 12 --slots 16 \
  --mixed-slots --threads 2 --eval-every 150 --lr .0007 --out runs/upc10m
```

`--resume` is a weight warm-start, not optimizer/sampler resume. `--warmup` warms quantization. The local pilot used no warmup. CUDA can be selected with `--device cuda`, but GPU execution was not tested in this experiment.

```python
import numpy as np
from mamba_ssm.models.gts_upc import GTS3UPC
from mamba_ssm.models.upc_frontend import compile_policy
from mamba_ssm.models.upc_runtime import Runtime, Decoder

model = GTS3UPC.load('runs/upc10m/checkpoint.pt').freeze()
policy = compile_policy('emit(-abs(state("x")-0.5))')
action = Runtime(Decoder(model)).act(
    policy, {'x': np.array([0.1, 0.5, 0.8], np.float32)})
print(action)
```

`Decoder()` without a model is the exact baseline. `Decoder(model)` uses neural predictions without teacher correction. Failed reads/writes remain visible. `upc_install.install` decodes static instructions once; `Runtime(Decoder())` can subsequently execute the returned IR. Cache identities include the actual model-decoded IR.

Temperature applies only to legal final action selection, not arithmetic, addresses or control-flow semantics. The interpreter bounds symbols, instructions, tensors and broadcasts; the restricted parser never evaluates Python. This is not a security audit or a production multi-tenant sandbox.

## Recorded pilot

The experiment archive delivered with this change includes the trained float master and 2-bit storage checkpoint, sparse C backbone, shape-specialized C executor, fixed game policies/adapters, tests and raw logs. Checkpoints and the CPU deployment/game harness are not committed in this core-only draft.

Local optimization used a **reduced standalone implementation of the inspected GTS equations**, not a byte-identical copy of the original module. The original package was not mounted locally. The repository-native wrapper imports actual `GTSMixed`, but original-package numerical equivalence and its new test file still need to be run there. The reported standalone suite passed 29 tests and skipped the optional upstream-equivalence test. Do not report that as a completed upstream test run.

One seed, 900 steps, 10,800 synthetic instruction/binding records, working sets of 4/8/16/32 slots. No English, game data, rewards or demonstrations. Elapsed through step 900: 342.7 seconds including periodic validation, two CPU threads on a Xeon Platinum 8573C.

| Fresh test working set | Complete opcode + four-pointer accuracy |
|---|---:|
| 4 / 8 / 16 / 32 slots | 256/256 at each size |
| 64 slots, absent from training | 253/256 |

On 256 sixteen-slot records, bypassing all GTS blocks or randomizing weights reduces complete-effect accuracy to zero. Three repeated-loop checks (4/32/128 iterations) complete correctly. These are narrow binding/execution controls, not general reasoning benchmarks.

Frozen neural interpretation and neural-install-plus-C matched the exact policy on three short episodes per game (100 steps/pieces): Pong 2 mean returns and zero misses, Tetris 36.67 mean lines, lane shooter 12.67 mean kills. There are only six Pong returns total per controller. Tetris exposes legal hard-drop afterstates, and the shooter is a small structured-state lane game, not an FPS. The policies are hand-authored.

## Performance boundary

One 21-token effect query: dense Torch median 22.122 ms; sparse C backbone plus Torch input/head bridge median 2.935 ms. Single-threaded CPU; warm runtime. The C backbone uses float32 on ternary-valued tables, not packed integer arithmetic.

C/Torch typed decisions agree on 128/128 probes, but maximum absolute logit difference is 2.148. **Strict numerical parity is unresolved**, notwithstanding agreement on sampled argmax effects.

| Complete decision | Live neural interpretation | Installed exact C |
|---|---:|---:|
| Pong | 66.63 ms | 57.3 microseconds |
| Tetris | 81.03 ms | 112.6 microseconds |
| Lane shooter | 109.56 ms | 67.9 microseconds |

These are medians of per-episode medians. **Installed C performs no neural inference during ticks.** Neural decoding and shape compilation are charged separately (about 0.07-0.59 s in these runs; changed shapes may recompile). Compilation is not a neural advantage: an exact decoder could emit the same C.

The 2-bit checkpoint is 3,872,315 bytes on disk, expanded for computation. This is not a runtime RAM or integer-kernel claim.

## Next requirement

Batch several effects per neural invocation and train state-dependent binding, subgoal selection and consequence prediction, rather than invoking a 10M model for every deterministic primitive. Establish original-package and C parity, then test held-out program families and longer games. Current implementation is infrastructure and a decoding pilot, not the full universal-control result.
