# Build a Looped MoE

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albertlieyingadrian/build-a-looped-moe/blob/main/notebooks/quickstart.ipynb)

A looped transformer runs the same middle layers several times. **Inkling 2 Mini** (2026) is a looped
mixture-of-experts model that goes one step further: it has **no single residual stream**. Its 24 layers are grouped
into blocks of 6; every attention and MoE sublayer reads its own learned mix of the finished blocks; layers 6-17 run
three times, and every pass **restarts from the same input**, so the previous pass reaches the next only through a
**gated message**. Its feed-forward layers route each token to 8 of 128 experts.

In this course you rebuild each of those ideas at toy scale, in plain PyTorch, on a laptop CPU, continuing from
[build-a-looped-transformer](https://github.com/albertlieyingadrian/build-a-looped-transformer):

1. a recap: a plain looped GPT with one residual stream, and the chain puzzle (stage 01);
2. blocks and the active block, depth attention over blocks, separate q/k/v mixes, the output head's own mix
   (stages 02-05);
3. a sigmoid top-k router, a mixture of experts, load balancing through the router bias (stages 06-08);
4. the loop that restarts from the prelude every pass, the message bank, the gated message (stages 09-11);
5. a comparison against a plain loop and a standard transformer at matched active parameters, and lenses for a
   model with no single residual stream (stages 12-13);
6. an extension: Inkling 2 Mini's real 48-visit layout at toy width, checked logit for logit against the one-file
   [nanoloop-moe](https://github.com/albertlieyingadrian/nanoloop-moe) (stage 14).

This is an **Inkling-style** toy built from public sources. It is not affiliated with or endorsed by Thinking
Machines, and it is not their code.

The layout follows the stage-by-stage style of CodeCrafters' build-your-own-X courses (https://codecrafters.io): a
`course-definition.yml`, one description per stage in `stage_descriptions/`, a starter in `starter_templates/python/`,
the reference program after every stage in `solutions/python/`, and tests per stage. The model code follows the structure of nanoGPT. Nothing is copied
from either; both are credited below. This is an independent course, not affiliated with or endorsed by CodeCrafters.

## What is verified and what is a stand-in

Every architectural statement in the stage descriptions is checked against public sources: the `inkling2` and
`inkling2_loop` model code in Hugging Face transformers, the model card, the release post, and the nnsight post
(links below). Where the course simplifies, it says so.

| | Inkling 2 Mini (verified) | this course |
|---|---|---|
| layers, blocks | 24 layers, blocks of 6, boundaries at 0, 6, 12, 18 | 8 layers, blocks of 2 (the 24/6 layout in stage 14) |
| loop | layers 6-17, 3 passes, 48 visits; every pass restarts from the prelude's states | the same, core = layers 2-5 |
| block mixing | RMS-normalized blocks, learned zero-init query, softmax over blocks, `1/sqrt(d)` | the same |
| attention mixes | 3 queries (q, k, v) in full-attention layers, 1 in sliding-window layers | 3 queries in every layer (stand-in) |
| head | `final_input` mix, final norm, logit softcap 30 | the same |
| messages | 4 per core layer (q, k, v, MoE) read from the previous pass's core blocks by learned queries | the same |
| message gate | `(1 - tanh f) x + tanh m * msg`, rank-64 projection of `[norm x, norm msg]`; starts closed | the same, rank 4; closed by zero init |
| MoE | 128 experts, top 8, sigmoid scores, selection-only bias, renormalized x 8, softcapped SwiGLU, no shared expert | 4-16 experts, top 2-4, the same router; shared expert optional (off by default) |
| balancing | a selection bias buffer exists; its training rule is not public | DeepSeek-V3's sign rule (assumption) |
| attention | sliding window + full, relative position bias, log scaling, q/k clipping, short convolutions, gated output | plain causal attention with rotary positions (stand-in) |

## Stages

| # | Stage | Command | What you learn |
|---|---|---|---|
| 01 | [Recap: a plain looped GPT](stage_descriptions/base-01-recap.md) | `recap` | prelude / core x K / coda with one residual stream; the chain puzzle |
| 02 | [Blocks and the active block](stage_descriptions/base-02-blocks.md) | `blocks` | freezing blocks at boundaries; their sum is the residual stream |
| 03 | [Depth attention over blocks](stage_descriptions/base-03-mix.md) | `mix` | a learned softmax mix of normalized blocks; zero query = average |
| 04 | [Separate q/k/v mixes](stage_descriptions/base-04-qkv.md) | `qkv` | attention's queries, keys and values read different blends |
| 05 | [The output head's own mix](stage_descriptions/base-05-final-mix.md) | `final-mix` | `final_input`, and what it reads at initialisation |
| 06 | [A sigmoid top-k router](stage_descriptions/base-06-router.md) | `router` | sigmoid scores, top k, route_scale, a bias that only picks |
| 07 | [A mixture of experts](stage_descriptions/base-07-moe.md) | `moe` | stacked experts, softcapped SwiGLU, stored vs active parameters |
| 08 | [Load balancing](stage_descriptions/base-08-balance.md) | `balance` | auxiliary-loss-free balancing through the router bias |
| 09 | [Loop it: every pass restarts](stage_descriptions/base-09-loop.md) | `loop` | why K does nothing without messages; 48 visits |
| 10 | [The message bank](stage_descriptions/base-10-bank.md) | `bank` | the previous pass's core blocks, four messages per layer |
| 11 | [The gated message](stage_descriptions/base-11-gate.md) | `gate` | a low-rank tanh gate that starts closed |
| 12 | [Looped MoE vs plain loop](stage_descriptions/base-12-compare.md) | `compare` | matched active parameters, a K sweep |
| 13 | [Lenses without a residual stream](stage_descriptions/base-13-lens.md) | `lens` | logit lens on the active block vs the mix lens vs each visit's own mix |
| 14 | [The Inkling 2 Mini layout](stage_descriptions/ext-14-layout.md) (extension) | `layout` | 48 visits; identical logits to the one-file model |

## How to run

Requirements: Python 3.10+, PyTorch 2.x (`pip install torch`), and `pip install -r requirements.txt` (numpy,
matplotlib, pytest, ruff). Everything runs on a CPU, with no downloads.

```bash
cp -r starter_templates/python/code my-loopmoe          # your working copy; start with app/main.py
make test STAGE=01 DIR=my-loopmoe                       # one stage's tests against your code
make test-upto STAGE=08 DIR=my-loopmoe                  # every stage so far
make test-solutions                                     # every reference solution against its stage (what CI runs)
make run-all                                            # the commands behind the numbers in stages 12 and 13
make solutions                                          # regenerate solutions/python/01-13 from 14
```

Stuck? `solutions/python/NN-<slug>/code/` is the program right after stage NN. Stages 01-11 test in about a second
each (stage 01 trains for a few seconds); stages 12-13 train two small models once, about a minute on a laptop.

`reference/nanoloop_moe.py` is a copy of the one-file model from
[nanoloop-moe](https://github.com/albertlieyingadrian/nanoloop-moe), kept in sync with it; stage 14's test checks the
last solution against it.

## Limitations

- Toy scale: tens of thousands of parameters and one synthetic task. The default sizes are chosen so the tests run in
  seconds, which is too small for any model to pass the chain task's first plateau; stage 12 gives a larger command
  and its numbers. The one-file nanoloop-moe has GPU-scale runs.
- Several parts of Inkling 2 Mini are replaced by simpler stand-ins (the table above). The course teaches the
  block-mixing, MoE and message mechanics, not Inkling's full attention design or its training recipe.
- One seed per number unless stated.

## Related

- [nanoloop-moe](https://github.com/albertlieyingadrian/nanoloop-moe): this model in one file, with GPU comparisons.
- [build-a-looped-transformer](https://github.com/albertlieyingadrian/build-a-looped-transformer): the course before
  this one.
- [nanoloop-transformer](https://github.com/albertlieyingadrian/nanoloop-transformer): a plain looped transformer in
  one file.

## Credits and further reading

**Inkling 2 Mini**

- Inkling 2 Mini model card, https://huggingface.co/thinkingmachines/Inkling-2-Mini: 26.4B parameters, 3.4B active,
  24 layers, 8 of 128 experts per token, the middle 12 layers run three times.
- Inkling 2 Mini release post, https://thinkingmachines.ai/news/inkling-2-mini/: the looping layout, attention
  residuals, messages between passes and a gate that starts closed.
- The `inkling2` and `inkling2_loop` model code in Hugging Face transformers, https://github.com/huggingface/transformers:
  the source for every "verified" line above.
- Opening the box of Inkling 2 Mini (nnsight blog),
  https://nnsight.net/blog/2026/11/20/opening-the-box-of-inkling-2-mini-using-nnsight-ndif/: block lists, the active
  block, the message bank and gates, with code to read them.
- The Looped Lens Challenge (NDIF), https://ndif.us/looped-lens-challenge/: build a lens for Inkling 2 Mini; the
  question stage 13 turns into a toy.

**Block mixing**

- Attention Residuals (2026), https://arxiv.org/abs/2603.15031: depth attention over earlier blocks, the design
  Inkling 2 Mini follows.
- DenseFormer (2024), https://arxiv.org/abs/2402.02622: each block reads a learned weighted average of all earlier
  outputs.
- Hyper-Connections (2024), https://arxiv.org/abs/2409.19606: learned connections across depth in place of the single
  residual.

**Looped and recurrent-depth models**

- Universal Transformers (2018), https://arxiv.org/abs/1807.03819.
- Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn, 2025),
  https://arxiv.org/abs/2502.05171.

**Mixture of experts**

- GShard (2020), https://arxiv.org/abs/2006.16668: top-2 routing and an auxiliary balancing loss.
- Switch Transformers (2021), https://arxiv.org/abs/2101.03961: top-1 routing at scale.
- DeepSeekMoE (2024), https://arxiv.org/abs/2401.06066: fine-grained and shared experts.
- DeepSeek-V3 Technical Report (2024), https://arxiv.org/abs/2412.19437: sigmoid routing and auxiliary-loss-free
  balancing through a selection bias (stage 08's rule).

**The structure of this course and its code**

- nanoGPT, https://github.com/karpathy/nanoGPT.
- CodeCrafters' open-source course repos (MIT), https://github.com/codecrafters-io/build-your-own-tokenizer: the course
  layout.

## License

MIT. See [LICENSE](LICENSE).
