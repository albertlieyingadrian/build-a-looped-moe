In this stage, you'll build the experts and count stored versus active parameters.

### The idea

Each expert is a small SwiGLU feed-forward network. Inkling's is **softcapped**: `down(softcap(g, 4) * sigmoid(g) *
softcap(up, 25))`, where `g` is the gate projection, the sigmoid sees the raw gate and the caps
(`tanh(x / c) * c`) only bound the linear factors. Expert width in Inkling 2 Mini is 1,152, half the hidden size of
2,304 (all **verified** in the config and code).

Stack the E experts' weights into tensors of shape (E, D, H) and (E, H, D). The output for a token is the routing-
weighted sum of its chosen experts' outputs.

```python
g  = einsum("nd,edh->neh", x, w_gate)                       # every expert, every token
h  = softcap(g, 4) * sigmoid(g) * softcap(einsum("nd,edh->neh", x, w_up), 25)
out = einsum("neh,ehd,ne->nd", h, w_down, w)                # w is 0 for unchosen experts
```

At toy size it is faster to run every expert on every token and multiply the unchosen ones by exactly 0 than to
gather each expert's tokens. The result is identical; the command checks it against a token-by-token loop. A real
MoE gathers, because running 128 experts to keep 8 would waste 94% of the compute.

### Stored versus active

"26.4B total, 3.4B active" (the model card) means: the weights stored, versus the weights one token touches. With E
experts and k chosen, an MoE layer stores E experts but each token uses k of them. Our toy layer (8 experts of
1,536 parameters, top 2) stores 12,544 parameters and activates 3,328.

A **shared expert** sees every token (DeepSeekMoE's idea). Inkling 2 Mini's config has `n_shared_experts = 0`; the
option is here (`--n-shared 1`) so you can try it, but the faithful setting is 0.

```bash
$ ./your_program.sh moe
stacked experts = routing token by token: True
one MoE layer: 12544 stored parameters, 3328 active per token
one expert: 1536 parameters; shared experts: 0
a dense SwiGLU with the same active width: 3072 parameters
```

The dense SwiGLU of width `(top_k + n_shared) * d_expert` is the baseline stage 12 uses for matched active
parameters (the MoE's extra 256 active parameters are its router).

### Tests

The equality must be `True`, and the counts must follow `E * 3 * D * H + D * E` stored and `k * 3 * D * H + D * E`
active, with and without a shared expert.

### Check yourself

1. Where does the factor 3 in `3 * D * H` come from?
2. Why does `count_params` subtract idle experts but never the router?
