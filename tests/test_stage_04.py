"""Stage 04: qkv. Layer 0 has one mix; later layers have a 3-query attention mix and an MoE mix."""

from conftest import line_value, run


def test_qkv(tmp_path):
    out = run(["qkv"], cwd=tmp_path)
    assert line_value(out, "layer 0:").startswith("32 mixing parameters")
    assert line_value(out, "layer 1:").startswith("128 mixing parameters")
    assert line_value(out, "zero queries: q, k and v read the same input:") == "True"
    assert line_value(out, "trained queries: q, k and v read different inputs:") == "True"
