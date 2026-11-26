In this stage, you'll let each sublayer read a learned mix of the blocks instead of their sum.

### The idea: depth attention

Ordinary attention lets a token choose which *earlier tokens* to read. **Depth attention** lets a sublayer choose which
*earlier blocks* to read, at the same token. Inkling 2 Mini follows the Attention Residuals design (arXiv:2603.15031):

1. stack the finished blocks and the active block: N + 1 vectors per token;
2. RMS-normalize each one (no learned scale), so a block's size does not decide how much it counts;
3. score each against a learned **query** vector `q` (one per sublayer, shared by all tokens), divided by `sqrt(D)`;
4. softmax over the N + 1 blocks, and return the weighted average of the **normalized** blocks.

```python
stack = rms(cat([blocks, active[..., None, :]], dim=-2))     # (B, T, N+1, D)
w = softmax(stack @ q / sqrt(D), dim=-2)                    # (B, T, N+1)
x = (w[..., None] * stack).sum(-2)                          # the sublayer's input
```

An analogy: a reader with a stack of notebook pages and a question in mind. They skim every page, weigh each by how
well it answers their question, and read a blend.

### Worked example

With a zero query every score is 0, so every weight is `1 / (N + 1)`: a plain average. Point the query at block 1
(`q = 3 * rms(block 1)`, D = 32) and block 1 scores `3 * 32 / sqrt(32) ≈ 17` while the others score near 0: its
weight is about 1.

```bash
$ ./your_program.sh mix
weights with a zero query: 0.25 0.25 0.25 0.25
weights with the query pointed at block 1: 0.00 1.00 0.00 0.00
output change after scaling block 0 by 100: 0.000000
```

The last line is the normalization at work: a block 100 times larger reads the same.

**Verified against the public implementation** (the `inkling2` code in Hugging Face transformers): unweighted RMS norm
on the stacked states, zero-initialized queries, `1/sqrt(hidden)` scaling, normalized states as the values.

### What to build

`depth_attention(query, stack)` and `DepthMix(c, n_query=1)` with `weights(blocks, active)` (returns the normalized
stack and the weights) and `forward(blocks, active)` (returns a list of `n_query` mixes). Keep `mix=False` returning
the sum, for the ablation in stage 12.

### Tests

The zero-query weights must all be 0.25, the pointed query must put at least 0.99 on block 1, and scaling a block
must change the output by less than 1e-5.

### Check yourself

1. Because the values are normalized, can a mix ever have a larger norm than `sqrt(D)`?
2. What does a sublayer lose compared with the residual stream when every block is normalized?
