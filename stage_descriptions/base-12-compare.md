In this stage, you'll train the looped MoE against a plain loop and a standard transformer.

### The fight

Four models, all with the same width and layer layout, the dense ones with an FFN as wide as the MoE's active
experts, so they touch about the same number of parameters per token:

| name | what it is |
|---|---|
| `standard` | one residual stream, no loop (K = 1), dense SwiGLU |
| `plainloop` | the looped course's recipe: one stream, core x K with input injection, dense |
| `loopmoe_k1` | this course's model trained with K = 1 (no loop) |
| `loopmoe` | the full model: block mixing, MoE, messages, K = 3 |

The command trains each on the chain task with the hop curriculum, sweeps K from 1 to `--k-max` and writes
`compare_s<seed>.json`. The one-file nanoloop-moe runs the same comparison at a larger size (and a fifth model,
`loopmoe_nomix`) on a GPU; its README has those numbers.

### Reading the numbers

At the default toy size (32 wide, 600 steps) every model sits at about 0.55: the plateau where it guesses one of
the two digits in the prompt. That is the honest result of a run that fits in a test. To see the curriculum climb,
give it width and steps, for example on a laptop:

```bash
$ ./your_program.sh compare --n-embd 64 --batch-size 64 --lr 1e-3 --steps 3000 --hops 3 --verbose
```

`run_all.sh` has the exact command behind the table below.

RESULTS_PLACEHOLDER

### What to build

`train(c)`: AdamW, gradient clipping at 1.0, the curriculum, `balance()` after every step. `compare` builds the
four configs, trains, sweeps and saves.

### Tests

A short run (`--steps 60`) must write a JSON with all four models, a sweep per model with accuracies in [0, 1],
equal stored and active counts for the dense models, and fewer active than stored parameters for the MoE ones.

### Check yourself

1. `loopmoe_k1` and `loopmoe` have the same weights. Which one gets more compute per token, and how much more?
2. Why compare at matched *active* parameters rather than matched stored parameters?
