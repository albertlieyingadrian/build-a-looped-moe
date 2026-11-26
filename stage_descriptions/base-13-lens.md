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

LENS_PLACEHOLDER

### Tests

Every lens list must have one entry per visit plus `out`, every value must be in [0, 1], and the `mix` value at
`out` must equal the model's own accuracy.

### Check yourself

1. At the first visit of pass 2, the active block is the prelude's. What does `act_in` read there, and why is that
   the same at every pass?
2. Design a fourth lens: a learned translator per visit (like a tuned lens). What would you train it to predict?
