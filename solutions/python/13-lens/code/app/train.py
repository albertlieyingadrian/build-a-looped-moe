"""Stage 01 (and 08, 12): the training loop on the chain task, with a hop curriculum and router balancing."""

import random
import time
from dataclasses import replace

import torch

from app.chain import CHAIN_VOCAB, chain_batch
from app.model import build, count_params


def train(c, log=print):
    """Train a model on chains; one more hop is unlocked each time train accuracy passes 85%."""
    torch.manual_seed(c.seed)
    rng = random.Random(c.seed)
    c.vocab = CHAIN_VOCAB
    model = build(c)
    total, active = count_params(model)
    log(f"{c.arch}: {total:,} params, {active:,} active | K={c.loops} mix={c.mix} msgs={c.messages} moe={c.moe}")
    opt = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=0.0)
    moes = [m for m in model.modules() if hasattr(m, "balance")]
    level, acc, t0, losses = 1, 0.0, time.time(), []
    for step in range(1, c.steps + 1):
        x, y, _ = chain_batch(rng, replace(c, hops=level))
        K = rng.randint(1, c.loops) if c.random_loops else c.loops
        logits, loss = model(x, K, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        for m in moes:
            m.balance(c.bias_rate)
        losses.append(loss.item())
        scored = y >= 0
        acc = 0.95 * acc + 0.05 * (logits.argmax(-1)[scored] == y[scored]).float().mean().item()
        if acc > 0.85 and level < c.hops:
            level, acc = level + 1, 0.0
            log(f"step {step:5d} | curriculum: now up to {level} hops")
        if step % max(1, c.steps // 5) == 0 or step == 1:
            msg = f"step {step:5d} | loss {loss.item():.3f} | train acc {acc:.2f} at <={level} hops"
            log(f"{msg} | {time.time() - t0:.0f}s")
    model.losses, model.level = losses, level
    return model
