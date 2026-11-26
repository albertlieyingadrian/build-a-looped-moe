"""Stage 14 (extension): the Inkling 2 Mini layout, and logits identical to the one-file reference model."""

import importlib.util
import re

import torch

from conftest import ROOT, run


def reference():
    spec = importlib.util.spec_from_file_location("nanoloop_moe", ROOT / "reference" / "nanoloop_moe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_layout(tmp_path):
    out = run(["layout"], cwd=tmp_path)
    rows = re.findall(r"visit\s+(\d+): layer\s+(\d+) .*mixes: (\d+)", out)
    assert len(rows) == 48
    n = {int(v): int(b) for v, _, b in rows}
    assert n[17] == 3 and n[18] == 2, "pass 2 restarts from the prelude's blocks"
    assert n[47] == 4


def test_matches_reference(tmp_path):
    nm = reference()
    from dataclasses import asdict

    for kw in ({}, {"n_shared": 1, "loops": 2}):
        c = nm.Config(vocab=nm.CHAIN_VOCAB, n_embd=32, n_head=2, n_experts=4, top_k=2, d_expert=16, gate_rank=4, **kw)
        torch.manual_seed(1)
        model = nm.LoopMoE(c).eval()
        for p in model.parameters():  # move every weight off its initial value, gates and queries included
            torch.nn.init.normal_(p, std=0.2)
        idx = torch.randint(0, len(c.vocab), (2, 24))
        with torch.no_grad():
            ref = model(idx, c.loops)[0]
        state = {"config": asdict(c), "model": model.state_dict(), "input": idx, "loops": c.loops}
        torch.save(state, tmp_path / "state.pt")
        run(["layout", "--state", tmp_path / "state.pt", "--logits", tmp_path / "logits.pt"], cwd=tmp_path)
        mine = torch.load(tmp_path / "logits.pt")
        assert torch.allclose(mine, ref, atol=1e-5), f"max difference {(mine - ref).abs().max():.2e}"
