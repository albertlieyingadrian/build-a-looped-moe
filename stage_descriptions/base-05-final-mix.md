In this stage, you'll add `final_input`, the mix the output head reads.

### The idea

With no single stream, the output head also has to choose what to read. Inkling 2 Mini gives it its own depth-attention
mix, `final_input`, over all the finished blocks and the last active block, followed by the final norm and the
unembedding (and a softcap of 30 on the logits).

At initialisation the query is zero, so the head reads a plain average of the normalized blocks, the embeddings
included. Training then learns which blocks the answer is in.

```
finished: [embed, prelude, core block A, core block B]   active: coda block
                     └──────── final_input mix ────────┘
                                   │
                           norm ──► head ──► softcap(30) ──► logits
```

This mix matters for interpretability: stage 13 uses it as a **lens**, applying it to the blocks available at an
earlier visit to ask "what would the model say if it stopped here?". The Looped Lens Challenge notes found that this
read is weak mid-loop on the real model, because the head's query was trained for the end of the network.

### Worked example

Our toy layout has 8 layers in blocks of 2: boundaries at layers 0, 2, 4 and 6, so the head mixes 4 finished blocks
plus the active coda block, 5 weights of 0.2 each at initialisation. Inkling 2 Mini's 24 layers in blocks of 6 give
the same 4 + 1.

```bash
$ ./your_program.sh final-mix
layers: 8, blocks of 2; the head mixes 4 finished blocks + the active one
final_input weights at initialisation: 0.20 0.20 0.20 0.20 0.20
logits = head(norm(final_input mix)): True
```

### What to build

`LoopMoE(c)`: embedding, `n_layer` `Layer`s, `final_input = DepthMix(c)`, `norm`, `head`. For now run the layers in
order once (the loop arrives in stage 09). Let `forward(idx, loops, targets, rec)` append one record per visit to
`rec` if given: `{"act_in": h entering the visit, "mlp_in": the MoE input, "blocks": the blocks it mixed}`, plus a
final `{"act_in", "blocks"}` record. Stage 13 reads them.

### Tests

The command must report 4 finished blocks, five weights of 0.20 and `True`.

### Check yourself

1. Why might a model trained with `final_input` put almost no weight on the embedding block?
2. If you read the model at visit 3 by applying `final_input` to the blocks that exist then, which blocks are missing?
