"""Stage 13: read every layer visit. With no single residual stream there are several things to read."""

import random

import torch

from app.chain import chain_batch


@torch.no_grad()
def lens(model, c, loops=None, n=500, seed=0):
    """Accuracy at the '?' positions of each reading, at each visit; the last entry is the model's real output.

    loopmoe:  act_in  the active block entering the visit, unembedded (the plain logit lens)
              mlp_in  the visit's own MoE-input mix of the blocks, message gated in
              mix     the output head's final_input mix over the current blocks + active block (the "mix lens")
    plainloop/standard: act_in, the one residual stream entering each core and coda visit."""
    model.eval()
    K = loops or c.loops
    x, y, _ = chain_batch(random.Random(seed), c, n=n)
    mask, rec = y >= 0, []
    model(x, K, rec=rec)
    gold = y[mask]

    def read(s):
        return (model.head(model.norm(s)).argmax(-1)[mask] == gold).float().mean().item()

    if c.arch != "loopmoe":
        labels = [f"L{i} p{p + 1}" for p in range(K) for i in range(model.core_start, model.core_end)]
        labels += [f"L{i}" for i in range(model.core_end, c.n_layer)] + ["out"]
        return {"visits": labels, "act_in": [read(r["act_in"]) for r in rec]}
    out = {"visits": [], "act_in": [], "mlp_in": [], "mix": []}
    looped = range(model.core_start, model.core_end)
    for (layer, p), r in zip(model.visits(K) + [("out", 0)], rec):
        out["visits"].append(f"L{layer} p{p + 1}" if layer in looped else f"L{layer}")
        out["act_in"].append(read(r["act_in"]))
        out["mlp_in"].append(read(r["mlp_in"]) if "mlp_in" in r else float("nan"))
        out["mix"].append(read(model.final_input(r["blocks"], r["act_in"])[0]))
    _, w = model.final_input.weights(rec[-1]["blocks"], rec[-1]["act_in"])
    out["final_weights"] = [round(v, 3) for v in w[mask].mean(0)[:, 0].tolist()]
    return out
