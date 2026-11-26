"""Stage 12: compare. Four models at matched active parameters, swept over K."""

from conftest import load_json


def test_compare(trained):
    res = load_json(trained / "compare_s0.json")
    assert set(res) == {"standard", "plainloop", "loopmoe_k1", "loopmoe"}
    for name, r in res.items():
        assert set(r["sweep"]) == {"1", "2", "3"}
        assert all(0 <= a <= 1 for accs in r["sweep"].values() for a in accs.values())
        if name in ("standard", "plainloop"):
            assert r["params"] == r["active"]
        else:
            assert r["active"] < r["params"]
    assert res["loopmoe"]["params"] == res["loopmoe_k1"]["params"]
    assert (trained / "loopmoe.pt").exists() and (trained / "plainloop.pt").exists()
