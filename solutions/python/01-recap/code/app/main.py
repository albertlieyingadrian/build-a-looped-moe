"""Build a looped MoE: one command per stage.

    ./your_program.sh <command> [options]

Stage 01 recap   02 blocks   03 mix   04 qkv   05 final-mix   06 router   07 moe   08 balance
      09 loop    10 bank     11 gate  12 compare  13 lens
Extension: 14 layout (the Inkling 2 Mini layout, 48 visits, and a check against the one-file model)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

MAX_STAGE = 1  # commands from later stages are hidden; the staged solutions lower this number

torch.set_num_threads(4)


def out_path(args, name):
    path = Path(args.out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path / name


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1))
    print(f"wrote {path}")


def small(**kw):
    """The toy size every command uses: fast on a laptop CPU."""
    from app.chain import CHAIN_VOCAB
    from app.model import Config

    base = dict(
        vocab=CHAIN_VOCAB, n_embd=32, n_head=2, block_layers=2, n_experts=4, top_k=2, d_expert=16, gate_rank=4,
        hops=3, block_size=40, batch_size=32, steps=300, lr=2e-3, seed=0,
    )  # fmt: skip
    return Config(**{**base, **kw})


def random_ids(c, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, len(c.vocab), (B, c.block_size), generator=g)


# ------------------------------------------------------------------------------------------------ stage 01: recap
def cmd_recap(args):
    from app.model import build, n_params
    from app.train import train

    c = small(arch="plainloop", moe=False, hops=2, steps=args.steps, seed=args.seed)
    torch.manual_seed(args.seed)
    model = build(c)
    for K in (1, 3, 6):
        print(f"K={K}  parameters {n_params(model)}  layer visits {len(model.visits(K))}")
    model = train(c, log=lambda *_: None)
    first, last = sum(model.losses[:20]) / 20, sum(model.losses[-20:]) / 20
    print(f"loss over the first 20 steps: {first:.3f}")
    print(f"loss over the last 20 steps: {last:.3f}")


COMMANDS = {
    "recap": (1, cmd_recap),
}


def parser():
    p = argparse.ArgumentParser(
        prog="your_program.sh", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)
    for name, (stage, _) in COMMANDS.items():
        if stage > MAX_STAGE:
            continue
        s = sub.add_parser(name)
        s.add_argument("--seed", type=int, default=0)
        s.add_argument("--out-dir", default="runs")
        if name == "recap":
            s.add_argument("--steps", type=int, default=300)
        if name == "moe":
            s.add_argument("--n-shared", type=int, default=0)
        if name == "balance":
            s.add_argument("--steps", type=int, default=300)
            s.add_argument("--rate", type=float, default=0.01)
        if name in ("compare", "lens"):
            s.add_argument("--steps", type=int, default=600)
            s.add_argument("--hops", type=int, default=3)
            s.add_argument("--n-embd", type=int, default=32)
            s.add_argument("--batch-size", type=int, default=32)
            s.add_argument("--lr", type=float, default=2e-3)
            s.add_argument("--k-max", type=int, default=4)
            s.add_argument("--only", default="")
            s.add_argument("--verbose", action="store_true")
        if name == "lens":
            s.add_argument("--n", type=int, default=300)
        if name == "layout":
            s.add_argument("--state", default="")
            s.add_argument("--logits", default="logits.pt")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    COMMANDS[args.command][1](args)


if __name__ == "__main__":
    sys.exit(main())
