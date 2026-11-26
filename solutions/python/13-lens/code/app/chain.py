"""Stage 01: the chain puzzle from nanoloop-transformer, and accuracy per number of hops."""

import random

import torch

CHAIN_VOCAB = ".abcdefghijklmnopqrstuvwxyz0123456789=;?"  # "." is padding


def chain_example(rng, max_hops, n_chains=2):
    """Two chains like a=7;b=a;c=b, interleaved and written defined-before-use, then one question per variable.
    'c?' needs 3 lookups (c->b->a->7). Returns the text and [(index of '?', answer digit, hops needed)]."""
    lens = [rng.randint(1, max_hops) for _ in range(n_chains)]
    names = iter(rng.sample("abcdefghijklmnopqrstuvwxyz", sum(lens)))
    chains, answers = [], {}
    for n in lens:
        vs, digit = [next(names) for _ in range(n)], str(rng.randrange(10))
        for i, v in enumerate(vs):
            answers[v] = (digit, n - i)
        chains.append([f"{v}={t}" for v, t in reversed(list(zip(vs, vs[1:] + [digit])))])
    statements = []
    while any(chains):
        statements.append(rng.choice([ch for ch in chains if ch]).pop(0))
    queries = list(answers)
    rng.shuffle(queries)
    text = ";".join(statements) + ";" + "".join(f"{q}?" for q in queries)
    start = len(text) - 2 * len(queries)
    return text, [(start + 2 * i + 1, *answers[q]) for i, q in enumerate(queries)]


def chain_batch(rng, c, n=None):
    """x: puzzles left-padded with '.'; y: the answer digit at every '?', -1 elsewhere; hops: lookups needed."""
    n = n or c.batch_size
    stoi = {ch: i for i, ch in enumerate(CHAIN_VOCAB)}
    x = torch.zeros(n, c.block_size, dtype=torch.long)
    y = torch.full((n, c.block_size), -1, dtype=torch.long)
    hops = torch.zeros(n, c.block_size, dtype=torch.long)
    for b in range(n):
        text, qs = chain_example(rng, c.hops)
        off = c.block_size - len(text)
        x[b, off:] = torch.tensor([stoi[ch] for ch in text])
        for i, digit, h in qs:
            y[b, off + i], hops[b, off + i] = stoi[digit], h
    return x, y, hops


@torch.no_grad()
def chain_accuracy(model, c, loops, n=1000, seed=0):
    """Accuracy per hop count on a fixed held-out set (same seed -> same puzzles)."""
    model.eval()
    dev = next(model.parameters()).device
    x, y, hops = chain_batch(random.Random(seed), c, n=n)
    pred = torch.cat([model(xb.to(dev), loops)[0].argmax(-1).cpu() for xb in x.split(250)])
    model.train()
    correct = pred == y
    return {h: correct[hops == h].float().mean().item() for h in range(1, c.hops + 1)}


def sweep(model, c, max_loops=6):
    """Chain accuracy for K = 1..max_loops: {K: {hop: acc}}."""
    return {K: chain_accuracy(model, c, K) for K in range(1, max_loops + 1)}
