"""Stage 13: lens. Three readings per visit for the looped MoE, one for the plain loop."""

from conftest import load_json, run


def test_lens(trained):
    run(["lens", "--n", "100", "--out-dir", trained], cwd=trained)
    res = load_json(trained / "lens.json")
    lm, pl = res["loopmoe"], res["plainloop"]
    n_visits = 2 + 3 * 4 + 2
    assert len(lm["visits"]) == len(lm["act_in"]) == len(lm["mix"]) == n_visits + 1
    assert len(pl["act_in"]) == 3 * 4 + 2 + 1
    for r in (lm, pl):
        assert all(0 <= v <= 1 for v in r["act_in"])
    assert abs(lm["mix"][-1] - lm["model_accuracy"]) < 1e-6, "the mix lens at the output is the model itself"
    assert abs(sum(lm["final_weights"]) - 1) < 1e-3
