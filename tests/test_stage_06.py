"""Stage 06: router. Sigmoid scores, top-k, weights renormalized to route_scale, bias only for choosing."""

from conftest import line_value, run


def test_router(tmp_path):
    out = run(["router"], cwd=tmp_path)
    assert line_value(out, "experts per token:") == "[2]"
    assert line_value(out, "weights sum to route_scale = 2.0:") == "True"
    assert line_value(out, "with bias +10 on expert 3, every token picks it:") == "True"
    assert line_value(out, "the bias picks experts but the weights still come from the scores:") == "True"
