In this stage, you'll loop the core, Inkling-style: every pass restarts from the prelude.

### The idea

The looped course carried the state from one pass into the next (`s = core(s + e)`). Inkling 2 Mini does not. Its
layers 0-5 are the **prelude**, 6-17 the **core**, 18-23 the **coda**. The core runs three times, and **every pass
starts again from the prelude's blocks**: the block list is reset to what it was after layer 5, and the active block
too. The two core blocks a pass built are thrown away; only a message survives (stages 10-11).

```
prelude ──► (blocks_P, h_P) ──► pass 1 ──► core blocks A1, B1
                │
                ├─────────────► pass 2 ──► A2, B2        (starts from blocks_P, h_P again)
                └─────────────► pass 3 ──► A3, B3 ──► coda ──► final_input ──► head
```

Analogy: three drafts of an essay, each written from the same notes on a fresh page. Without a way to see the
previous draft, the drafts come out identical.

That is exactly what this stage checks: with messages off, **K does nothing**. The output at K = 4 equals K = 1 to
the last bit.

**Verified** (transformers code, the release post and the nnsight post): 24 layers, core 6-17, 3 passes, 48 layer
visits numbered in the order they run, the reset to the prelude's states at every pass, a separate KV cache per
visit. In our toy layout (8 layers in blocks of 2) the core is layers 2-5.

```bash
$ ./your_program.sh loop
K=2: change in logits from K=1 without messages: 0.000000
K=3: change in logits from K=1 without messages: 0.000000
K=4: change in logits from K=1 without messages: 0.000000
Inkling 2 Mini layout: 24 layers, 48 layer visits at K=3
visit 18 = layer 6, pass 2; visit 42 = layer 18
```

### What to build

`LoopMoE.forward`: run the prelude layers, save `(h, blocks)`, then K times reset to them and run the core layers;
then the coda and `final_input`. `visits(K)` lists `(layer, pass)` for every visit.

### Tests

All three changes must be below 1e-5 and the layout line must say 24 layers and 48 visits.

### Check yourself

1. Why does the restart make the KV cache of each pass independent?
2. Without messages, is there any reason to run the core more than once?
