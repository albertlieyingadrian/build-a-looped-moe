In this extension stage, you'll build Inkling 2 Mini's real layout at toy width and check your model against the
one-file reference.

### The layout

24 layers in blocks of 6: prelude 0-5, core 6-17 run three times, coda 18-23. That is 48 layer visits. The command
prints, for every visit, how many finished blocks its MoE input mixes:

```bash
$ ./your_program.sh layout
visit  0: layer  0 once    finished blocks its MoE input mixes: 1
...
visit  6: layer  6 pass 1  finished blocks its MoE input mixes: 2
visit 12: layer 12 pass 1  finished blocks its MoE input mixes: 3
visit 18: layer  6 pass 2  finished blocks its MoE input mixes: 2
...
visit 42: layer 18 once    finished blocks its MoE input mixes: 4
```

The count drops from 3 back to 2 at visit 18: the reset to the prelude's blocks at the start of pass 2.

### Matching the one-file model

`reference/nanoloop_moe.py` in this repository is a copy of nanoloop-moe's one file, kept in sync with it. With
`--state`, the command loads a saved `{"config", "model", "input", "loops"}` file into your `LoopMoE`, runs it and
saves the logits. The test builds the reference model, saves its weights, runs your program on them and checks that
the logits agree to 1e-5. For that your parameter names must match the reference's (`layers.3.attn_input.query`,
`layers.3.gates.0.down.weight`, `message_input.query`, `final_input.query`, and so on): the reference solution's
names are the spec.

### Tests

The layout must have 48 visits with the reset at visit 18, and your logits must match the reference's.
