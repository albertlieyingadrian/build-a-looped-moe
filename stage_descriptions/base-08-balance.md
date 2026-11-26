In this stage, you'll keep every expert busy without adding a loss term.

### The idea

Left alone, a router tends to fall in love with a few experts: they get more tokens, train faster, score higher and
get even more tokens. Older MoEs (GShard, Switch) add an **auxiliary loss** that penalizes uneven load. DeepSeek-V3
introduced **auxiliary-loss-free balancing**: after each step, nudge each expert's selection bias up if it got fewer
tokens than average and down if it got more:

```python
bias += rate * sign(mean(load) - load)        # load = tokens routed to each expert this step
```

Because the bias only affects *selection* (stage 06), this steers traffic without pulling the loss in a second
direction. Think of a host seating guests: when one table is full, they steer new arrivals elsewhere, without
changing the food.

**What is verified and what is not.** Inkling 2 Mini's router has a per-expert `e_score_correction_bias` added only
when choosing experts (verified in the code). How that bias was trained is **not** in the public inference code; the
sign rule above is DeepSeek-V3's, and it is **our assumption** for Inkling.

### Worked example

The command builds 8 experts and 512 tokens that all lean the same way, so a few experts win. Each balance step
counts the load and nudges the bias by 0.01.

```bash
$ ./your_program.sh balance
load max/mean before balancing: 2.62
load max/mean after 300 balance steps: 1.15
router bias: +0.10 +0.21 +0.21 -0.20 +0.16 -0.15 -0.19 -0.13
```

A max/mean of 1.0 is perfectly even; 2.62 means the busiest expert did 2.6 times its share.

### What to build

`MoE.forward` adds the chosen experts' counts to `self.load` in training mode; `MoE.balance(rate)` applies the rule
and zeroes `load`. The training loop calls `balance(c.bias_rate)` on every MoE after every optimizer step.

### Tests

The imbalance after balancing must be below 1.4 and below the imbalance before.

### Check yourself

1. What happens with a `rate` that is too large?
2. Why use the sign of the difference rather than the difference itself?
