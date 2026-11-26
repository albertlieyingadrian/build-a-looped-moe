In this stage, you'll rebuild the model the looped course ended with: a plain looped GPT with one residual stream.

### The idea: one running total

In an ordinary transformer every sublayer adds its output to one running total, the **residual stream**. Think of a
shared whiteboard: each layer reads the whole board and writes a few more lines on it. A looped transformer sends
that board through the same middle layers K times:

```
embed ──► prelude ──► e ──┬──► core ──► core ──► ... (K times, same weights) ──► coda ──► head
                          └── re-added before every pass ("input injection")
```

Every later stage of this course changes one thing about this picture. By stage 11 there is no whiteboard at all.

### The task: follow a chain

The puzzle is nanoloop-transformer's chain task. Two chains of assignments are interleaved, written so every variable
is defined before it is used, and then every variable is asked about:

```
z=4;m=5;u=z;k=m;y=k;z?k?m?y?u?
```

`y?` needs 3 lookups (y → k → m → 5). Answers never appear after a `?`, so each must be worked out; a model resolves
roughly one more link per layer or per pass. Only the `?` positions are scored (the target is -1 elsewhere).

### What to build

- `Config`, a dataclass holding every size (the reference keeps the same field names all course long).
- `PlainLoop`: an embedding, `n_layer = block_layers * (prelude_blocks + core_blocks + coda_blocks)` pre-norm layers
  (RMSNorm, causal attention with rotary positions, a SwiGLU), a final norm and a head. The core runs K times with
  the prelude's output re-added before each pass. `visits(K)` lists `(layer, pass)` in the order they run.
- `chain_example`, `chain_batch` and a training loop with a hop curriculum (unlock one more hop when train accuracy
  passes 85%).

```bash
$ ./your_program.sh recap
K=1  parameters 60448  layer visits 8
K=3  parameters 60448  layer visits 16
K=6  parameters 60448  layer visits 28
loss over the first 20 steps: 2.891
loss over the last 20 steps: 0.636
```

### Tests

The parameter count must not depend on K, the number of visits must be `prelude + K * core + coda` layers, and the
loss over the last 20 of 300 steps must be lower than over the first 20.

### Check yourself

1. Why does the parameter count not change with K, while the compute grows linearly with it?
2. After 300 steps the accuracy sits near 0.55. What simple strategy gets about half of the answers right?
