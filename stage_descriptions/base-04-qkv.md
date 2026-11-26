In this stage, you'll give attention's queries, keys and values their own mixes of the blocks.

### The idea

A residual stream gives every sublayer the same input. With depth attention each *input* can differ. In Inkling 2
Mini the full-attention layers use **three queries** in their attention mix, one each for the attention's queries,
keys and values, and one more query for the MoE input. So a layer can ask "what should I look for" from one blend of
blocks and "what do I offer to others" from another.

```
blocks + active ──► mix_q ──► norm_q ──► W_q ┐
                ──► mix_k ──► norm_k ──► W_k ├─► causal attention ──► active block
                ──► mix_v ──► norm_v ──► W_v ┘
(after attention) blocks + active ──► mix_mlp ──► norm_mlp ──► MoE ──► active block
```

Layer 0 has no attention mix: its only input is the embeddings. Each norm has its own learned scale, which is why
stage 02 asked for four of them.

**Verified:** full-attention layers have `attention_input` with 3 queries and separate `q/k/v_input_layernorm`;
sliding-window layers have 1 query; layer 0 has no `attention_input`. **Stand-in in this course:** every layer uses
the 3-query version and full causal attention (Inkling alternates sliding-window and full layers; at toy sequence
lengths a 1,024-token window would see everything anyway).

### Worked example

With `n_embd = 32`: layer 0 has one mix (`mlp_input`, 32 parameters); every later layer has a 3-query attention mix
(96) and an MoE mix (32): 128 parameters. Tiny next to the attention weights (4 × 32 × 32 = 4,096).

```bash
$ ./your_program.sh qkv
layer 0: 32 mixing parameters (1 mixes)
layer 1: 128 mixing parameters (2 mixes)
zero queries: q, k and v read the same input: True
trained queries: q, k and v read different inputs: True
```

### What to build

`Layer` uses `DepthMix(c, n_query=3)` as `attn_input` (None for layer 0) and `DepthMix(c)` as `mlp_input`;
`Attention.forward(xq, xk, xv)` takes three inputs.

### Tests

The mixing-parameter counts must be `n_embd` and `4 * n_embd`, and both `True` lines must print.

### Check yourself

1. At initialization the three mixes are identical. What breaks the symmetry during training?
2. Why does the MoE mix read the blocks *after* the attention output joined the active block?
