In this stage, you'll build the router that sends each token to a few experts.

### The idea: a triage desk

A mixture of experts replaces one big feed-forward network with many small ones, the **experts**, and a **router**
that sends each token to only a few of them. Like a hospital triage desk: every patient is scored for every
specialist, and sees only the top few.

Inkling 2 Mini routes every token to 8 of 128 experts per layer. Its router, **verified** in the public
implementation, works like this:

1. `scores = sigmoid(W_router x)`: one independent score in (0, 1) per expert (not a softmax over experts);
2. pick the top k of `scores + bias`, where `bias` is a per-expert buffer used **only for choosing**;
3. the chosen experts' weights are their raw `scores`, renormalized to sum to 1, times `route_scale` (8 in Inkling 2
   Mini, which also picks 8 experts; we set `route_scale = top_k`).

The bias is the load-balancing knob of stage 08. Keeping it out of the weights means balancing changes *who* answers,
never *how loudly*.

### Worked example

Scores for 4 experts: `[0.9, 0.2, 0.7, 0.4]`, bias `[0, 0, 0, 0.5]`, top 2.
Choice by `scores + bias = [0.9, 0.2, 0.7, 0.9]`: experts 0 and 3. Weights from the scores alone:
`0.9 / 1.3 * 2 = 1.38` and `0.4 / 1.3 * 2 = 0.62`.

```bash
$ ./your_program.sh router
experts per token: [2]
weights sum to route_scale = 2.0: True
with bias +10 on expert 3, every token picks it: True
the bias picks experts but the weights still come from the scores: True
```

### What to build

`MoE.route(flat) -> (w, idx)`: `w` is an (N, E) matrix with exactly `top_k` nonzero entries per row summing to
`route_scale`; `idx` is the (N, k) chosen experts. Register `bias` (and `load`, for stage 08) as buffers, not
parameters: the optimizer must not touch them.

### Tests

Every token must have exactly 2 experts, the sums must match, and both bias lines must print `True`.

### Check yourself

1. With a softmax router, raising one expert's score lowers every other expert's. Does that happen with sigmoid scores?
2. Why would putting the bias into the weights fight against the loss?
