In this stage, you'll split the residual stream into finished blocks and the block being built.

### The idea: a notebook instead of a whiteboard

Inkling 2 Mini groups its 24 layers into **blocks of 6**. Inside a block, sublayer outputs add up as usual: that
running sum is the **active block**. At a block boundary the active block is **frozen** into a list of finished
blocks and a new active block starts from zero. Instead of one whiteboard, the model keeps a notebook: one page per
finished block, plus the page it is writing on.

```
layer:   0     1   |   2     3   |  ...
        [embed]    |  [block 0-1]       ← finished blocks (frozen at layers 0, 2, ...)
         active = attn0 + ffn0 + attn1 + ffn1   ← the active block, reset at every boundary
```

Layer 0 is a boundary too, so the first finished block is the token embeddings themselves.

A worked example with 4 layers and `block_layers = 2`:

| after layer | finished blocks | active block |
|---|---|---|
| 0 | [embed] | attn0 + ffn0 |
| 1 | [embed] | attn0 + ffn0 + attn1 + ffn1 |
| 2 | [embed, layers 0-1] | attn2 + ffn2 |
| 3 | [embed, layers 0-1] | attn2 + ffn2 + attn3 + ffn3 |

### Nothing has changed yet

Add up all the pages and you get the whiteboard back: `sum(finished blocks) + active block` is exactly the residual
stream of an ordinary transformer. So if every sublayer reads that sum, the block model *is* a standard transformer.
That is the `mix=False` setting of `DepthMix`, which this stage needs and the next stage replaces.

### What to build

`Layer(c, idx, looped)`, with `forward(h, blocks) -> (h, blocks)`. `h` is the active block (B, T, D) and `blocks`
is (B, T, N, D). At a boundary (`idx % block_layers == 0`) append the incoming `h` to `blocks` and restart the
active block with the attention output; otherwise add the attention output to `h`. The MoE / FFN output is always
added to `h`. The sublayers read `DepthMix(blocks, h)`, which for now returns `blocks.sum(-2) + h`. Keep separate
`norm_q`, `norm_k`, `norm_v` and `norm_mlp` (stage 04 explains why).

```bash
$ ./your_program.sh blocks
after layer 0: 1 finished blocks (boundary: True)
after layer 1: 1 finished blocks (boundary: False)
after layer 2: 2 finished blocks (boundary: True)
after layer 3: 2 finished blocks (boundary: False)
finished blocks + active block = the residual stream, max difference: 0.000000
```

The command builds 4 `Layer`s and 4 `PlainLayer`s with the same weights and compares them.

### Tests

The block counts must be 1, 1, 2, 2 and the difference from the plain stack below 1e-4.

### Check yourself

1. In Inkling 2 Mini, which layers are boundaries? How many finished blocks does layer 23 see?
2. The nnsight post on Inkling 2 Mini warns that `layers[i].output` is not "the state after layer i". Why not?
