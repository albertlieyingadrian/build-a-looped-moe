In this stage, you'll merge each message into the layer through a gate that starts closed.

### The idea

A message is not simply added. Each of the four inputs (q, k, v, MoE) has its own **message gate** that decides, per
number, how much of the current input to keep and how much of the message to let in:

```python
z = down(cat([rms(x), rms(msg)]))           # rank r: Inkling 2 Mini uses 64, our toy 4
g_forget  = tanh(forget(z))
g_message = tanh(message(z))
x = (1 - g_forget) * x + g_message * msg
```

**Verified:** that formula, the rank-64 `down_proj` on `[norm(x), norm(message)]`, and `tanh` for both gates
(transformers code and the nnsight post). The q, k and v gates act on the attention mixes before their norms; the MoE
gate acts on the MoE mix before `post_attention_layernorm`. The release post says the gate "initially blocks the
entire message", so each loop starts out behaving like a non-looped model. We get that by zero-initializing the
`forget` and `message` projections: `tanh(0) = 0`, so `x` passes through untouched.

Analogy: a new editor who at first changes nothing in the draft, and slowly learns which of the previous draft's
sentences to splice in.

### Worked example

A new model with messages on still gives identical logits at K = 1 and K = 3, because every gate is closed. Open the
gates (random `message` weights) and K starts to matter:

```bash
$ ./your_program.sh gate
a new gate passes x through unchanged: True
new model: change in logits from K=1 to K=3: 0.000000
gates opened: change in logits from K=1 to K=3: 1.743070
```

Training is what opens them: the gradient reaches the zero projections through `z`, which is not zero.

### What to build

`MessageGate(c)` and, in `Layer.__init__`, four gates for looped layers when `c.messages` is on. Apply them in
`Layer.forward` when messages are given.

### Tests

The first two lines must show `True` and a change below 1e-5, and the third a change above 1e-3.

### Check yourself

1. Why initialize both projections to zero rather than only `message`?
2. `g_forget` can be negative (tanh is in (-1, 1)). What does `1 - g_forget > 1` mean?
