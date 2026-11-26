In this stage, you'll read every layer visit, and find out why a model with no residual stream is hard to read.

### The idea

The **logit lens** takes the residual stream partway through a model and applies the final norm and head to it: "if
the model had to answer now, what would it say?" In a plain loop there is exactly one thing to read at a visit, the
stream.

In this model there is no stream. At a visit you can read at least three different things:

| lens | what it reads |
|---|---|
| `act_in` | the active block entering the visit (what `layers[i].input` is in Inkling 2 Mini) |
| `mlp_in` | the visit's own MoE-input mix of the blocks, with the message gated in |
| `mix` | the output head's `final_input` mix applied to the blocks that exist at this visit: the "mix lens" |

The active block holds only what the current block's layers wrote since its boundary, so the logit lens on it misses
everything frozen into earlier blocks. The mix lens reads everything, but with weights the head learned for the *end*
of the network. This is a toy version of the question behind NDIF's Looped Lens Challenge on Inkling 2 Mini, where
the lab notes found the active-block logit lens reads the answer mainly in the last block, the head's mix is no better
mid-loop, and each layer's own MoE-input mix is the best raw read.

### The `lens` command

It uses the `loopmoe.pt` and `plainloop.pt` checkpoints that `compare` saved in `--out-dir` (training them first if
they are missing) and prints the accuracy at the `?` positions for every visit and lens. The last row, `out`, is the
model's real output, so the mix lens there must equal the model's accuracy exactly.

### What we found (seed 0, the stage 12 checkpoints, 500 puzzles)

| visit | L0 | L1 | L2 p1 | L3 p1 | L4 p1 | L5 p1 | L2 p3 | L5 p3 | L6 | L7 | out |
|---|---|---|---|---|---|---|---|---|---|---|---|
| looped MoE, `act_in` (logit lens) | 0.10 | 0.26 | 0.20 | 0.48 | 0.54 | 0.35 | 0.20 | 0.38 | 0.52 | 0.46 | 0.42 |
| looped MoE, `mlp_in` | 0.25 | 0.24 | 0.47 | 0.53 | 0.53 | 0.54 | 0.45 | 0.54 | 0.55 | 0.56 | - |
| looped MoE, `mix` (mix lens) | 0.10 | 0.25 | 0.19 | 0.49 | 0.54 | 0.53 | 0.19 | 0.53 | 0.54 | 0.55 | **0.56** |
| plain loop, residual stream | | | 0.54 | 0.54 | 0.54 | 0.56 | 0.64 | 0.71 | 0.72 | 0.90 | **0.91** |

(Pooled accuracy over every `?`; the plain loop's prelude is not read.)

- **The plain loop's stream fills up pass by pass**: 0.54-0.56 in pass 1, 0.60-0.69 in pass 2, 0.64-0.71 in pass 3,
  then the coda finishes it (0.90). One residual stream, one obvious place to read.
- **The logit lens on the active block jumps around in the looped MoE.** At every block boundary (L2, L4, L6) the
  active block restarts, so it reads only the layer or two since then: 0.20 at L2 in every pass, and only 0.42 at the
  output, below the model's own 0.56, because the head never reads the active block alone.
- **The mix lens reads what the model would say, but adds little before the end**, and each visit's own MoE-input mix
  is the steadiest read (0.45-0.56). This matches, at toy scale and on a weak model, what the Looped Lens Challenge
  notes found on Inkling 2 Mini: the active-block logit lens reads late, the head's mix is no better mid-loop, and the
  per-layer MoE-input mix is the best raw read.
- The trained head's `final_input` weights stay close to uniform (0.16-0.23 over embed, prelude, two core blocks and
  the active block). This model has barely started to use its mixing; a longer run is the next thing to try.

### Tests

Every lens list must have one entry per visit plus `out`, every value must be in [0, 1], and the `mix` value at
`out` must equal the model's own accuracy.

### Check yourself

1. At the first visit of pass 2, the active block is the prelude's. What does `act_in` read there, and why is that
   the same at every pass?
2. Design a fourth lens: a learned translator per visit (like a tuned lens). What would you train it to predict?
