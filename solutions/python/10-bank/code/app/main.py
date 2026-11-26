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

MAX_STAGE = 10  # commands from later stages are hidden; the staged solutions lower this number

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


# ------------------------------------------------------------------------------------------------ stages 02-05: blocks
def cmd_blocks(args):
    from app.blocks import Layer
    from app.model import PlainLayer

    torch.manual_seed(args.seed)
    c = small(mix=False, moe=False, messages=False)
    layers = [Layer(c, i, looped=False) for i in range(4)]
    plain = [PlainLayer(c) for _ in range(4)]
    for lay, p in zip(layers, plain):  # give the two stacks the same weights
        for n in (lay.norm_q, lay.norm_k, lay.norm_v):
            n.load_state_dict(p.n1.state_dict())
        lay.norm_mlp.load_state_dict(p.n2.state_dict())
        lay.attn.load_state_dict(p.attn.state_dict())
        lay.ffn.load_state_dict(p.ffn.state_dict())
    x = torch.randn(2, 8, c.n_embd)
    h, blocks, ref = x, x.new_zeros(2, 8, 0, c.n_embd), x
    for i, (lay, p) in enumerate(zip(layers, plain)):
        h, blocks = lay(h, blocks)
        ref = p(ref)
        print(f"after layer {i}: {blocks.shape[-2]} finished blocks (boundary: {lay.boundary})")
    stream = blocks.sum(-2) + h
    print(f"finished blocks + active block = the residual stream, max difference: {(stream - ref).abs().max():.6f}")


def cmd_mix(args):
    from app.blocks import DepthMix
    from app.model import rms

    torch.manual_seed(args.seed)
    c = small()
    m = DepthMix(c)
    blocks, active = torch.randn(1, 1, 3, c.n_embd), torch.randn(1, 1, c.n_embd)
    _, w = m.weights(blocks, active)
    print("weights with a zero query: " + " ".join(f"{v:.2f}" for v in w[0, 0, :, 0].tolist()))
    with torch.no_grad():
        m.query[0] = 3 * rms(blocks[0, 0, 1])
    _, w = m.weights(blocks, active)
    print("weights with the query pointed at block 1: " + " ".join(f"{v:.2f}" for v in w[0, 0, :, 0].tolist()))
    out = m(blocks, active)[0]
    big = blocks.clone()
    big[..., 0, :] *= 100
    print(f"output change after scaling block 0 by 100: {(m(big, active)[0] - out).abs().max():.6f}")


def cmd_qkv(args):
    from app.blocks import Layer
    from app.model import n_params

    torch.manual_seed(args.seed)
    c = small(moe=False, messages=False)
    for i in (0, 1):
        lay = Layer(c, i, looped=False)
        mixes = [m for m in (lay.attn_input, lay.mlp_input) if m is not None]
        print(f"layer {i}: {sum(n_params(m) for m in mixes)} mixing parameters ({len(mixes)} mixes)")
    lay = Layer(c, 1, looped=False)
    blocks, active = torch.randn(2, 5, 1, c.n_embd), torch.randn(2, 5, c.n_embd)
    q, k, v = lay.attn_input(blocks, active)
    print(f"zero queries: q, k and v read the same input: {torch.allclose(q, k) and torch.allclose(k, v)}")
    torch.nn.init.normal_(lay.attn_input.query, std=1.0)
    q, k, v = lay.attn_input(blocks, active)
    print(f"trained queries: q, k and v read different inputs: {not torch.allclose(q, k) and not torch.allclose(k, v)}")


def cmd_final_mix(args):
    from app.loopmoe import LoopMoE

    torch.manual_seed(args.seed)
    c = small(moe=False, messages=False, loops=1)
    m = LoopMoE(c).eval()
    rec = []
    logits, _ = m(random_ids(c), rec=rec)
    blocks, h = rec[-1]["blocks"], rec[-1]["act_in"]
    _, w = m.final_input.weights(blocks, h)
    n = blocks.shape[-2]
    print(f"layers: {c.n_layer}, blocks of {c.block_layers}; the head mixes {n} finished blocks + the active one")
    print("final_input weights at initialisation: " + " ".join(f"{v:.2f}" for v in w[0, 0, :, 0].tolist()))
    from app.model import softcap

    direct = softcap(m.head(m.norm(m.final_input(blocks, h)[0])), 30.0)
    print(f"logits = head(norm(final_input mix)): {torch.allclose(direct, logits, atol=1e-5)}")


# ------------------------------------------------------------------------------------------------ stages 06-08: MoE
def cmd_router(args):
    from app.moe import MoE

    torch.manual_seed(args.seed)
    moe = MoE(small(n_experts=8, top_k=2))
    x = torch.randn(64, 32)
    w, idx = moe.route(x)
    print(f"experts per token: {sorted(set((w > 0).sum(-1).tolist()))}")
    sums = torch.allclose(w.sum(-1), torch.full((64,), moe.route_scale))
    print(f"weights sum to route_scale = {moe.route_scale}: {sums}")
    scores = torch.sigmoid(moe.router(x))
    with torch.no_grad():
        moe.bias[3] = 10.0
    w2, idx2 = moe.route(x)
    print(f"with bias +10 on expert 3, every token picks it: {bool((idx2 == 3).any(-1).all())}")
    chosen = scores.gather(1, idx2)
    expect = chosen / chosen.sum(-1, keepdim=True) * moe.route_scale
    same = torch.allclose(w2.gather(1, idx2), expect, atol=1e-5)
    print(f"the bias picks experts but the weights still come from the scores: {same}")


def cmd_moe(args):
    from app.model import count_params, n_params, softcap
    from app.moe import MoE

    torch.manual_seed(args.seed)
    c = small(n_experts=8, top_k=2, n_shared=args.n_shared)
    moe = MoE(c).eval()
    x = torch.randn(10, c.n_embd)
    w, idx = moe.route(x)
    ref = moe.shared(x).clone() if moe.shared is not None else torch.zeros_like(x)
    for n in range(10):
        for e in idx[n].tolist():
            g = x[n] @ moe.w_gate[e]
            h = softcap(g, 4.0) * torch.sigmoid(g) * softcap(x[n] @ moe.w_up[e], 25.0)
            ref[n] += w[n, e] * (h @ moe.w_down[e])
    print(f"stacked experts = routing token by token: {torch.allclose(moe(x), ref, atol=1e-5)}")
    total, active = count_params(moe)
    print(f"one MoE layer: {total} stored parameters, {active} active per token")
    print(f"one expert: {3 * c.n_embd * c.d_expert} parameters; shared experts: {c.n_shared}")
    dense = n_params(__import__("app.model", fromlist=["SwiGLU"]).SwiGLU(c.n_embd, (c.top_k + c.n_shared) * c.d_expert))
    print(f"a dense SwiGLU with the same active width: {dense} parameters")


def cmd_balance(args):
    from app.moe import MoE

    torch.manual_seed(args.seed)
    moe = MoE(small(n_experts=8, top_k=2))
    x = torch.randn(512, 32) + 2.0 * torch.randn(32)  # every token leans the same way: a few experts win
    imbalance = []
    for _ in range(args.steps + 1):
        moe.load.zero_()
        moe(x)
        imbalance.append((moe.load.max() / moe.load.mean()).item())
        moe.balance(args.rate)
    print(f"load max/mean before balancing: {imbalance[0]:.2f}")
    print(f"load max/mean after {args.steps} balance steps: {imbalance[-1]:.2f}")
    print("router bias: " + " ".join(f"{v:+.2f}" for v in moe.bias.tolist()))


# ------------------------------------------------------------------------------------------------ stages 09-11: loop
def cmd_loop(args):
    from app.loopmoe import LoopMoE

    torch.manual_seed(args.seed)
    c = small(messages=False)
    m = LoopMoE(c).eval()
    x = random_ids(c)
    base = m(x, 1)[0]
    for K in (2, 3, 4):
        print(f"K={K}: change in logits from K=1 without messages: {(m(x, K)[0] - base).abs().max():.6f}")
    ink = LoopMoE(small(block_layers=6, n_embd=16, messages=False))
    v = ink.visits(3)
    print(f"Inkling 2 Mini layout: {len(ink.layers)} layers, {len(v)} layer visits at K=3")
    print(f"visit 18 = layer {v[18][0]}, pass {v[18][1] + 1}; visit 42 = layer {v[42][0]}")


def cmd_bank(args):
    from app.loopmoe import LoopMoE

    torch.manual_seed(args.seed)
    c = small(loops=2)
    m = LoopMoE(c).eval()
    seen = []
    m.message_input.register_forward_hook(lambda mod, inp, out: seen.append((inp, out)))
    m(random_ids(c))
    (bank_blocks, bank_active, _), msgs = seen[0]
    print(f"message reads per forward at K=2: {len(seen)} (one per core layer of pass 2)")
    print(f"bank: {bank_blocks.shape[-2]} finished core blocks + 1 active block")
    print(f"messages per core layer: {len(msgs)}, each of shape {tuple(msgs[0].shape)}")
    from app.model import rms

    avg = rms(torch.cat([bank_blocks, bank_active.unsqueeze(-2)], -2)).mean(-2)
    print(f"zero queries: every message is the bank's average: {all(torch.allclose(x, avg, atol=1e-5) for x in msgs)}")


COMMANDS = {
    "recap": (1, cmd_recap),
    "blocks": (2, cmd_blocks),
    "mix": (3, cmd_mix),
    "qkv": (4, cmd_qkv),
    "final-mix": (5, cmd_final_mix),
    "router": (6, cmd_router),
    "moe": (7, cmd_moe),
    "balance": (8, cmd_balance),
    "loop": (9, cmd_loop),
    "bank": (10, cmd_bank),
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
