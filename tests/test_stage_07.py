"""Stage 07: moe. Stacked experts equal per-token routing; stored vs active parameter counts."""

from conftest import line_value, run

D, H, E, K = 32, 16, 8, 2


def counts(out):
    words = line_value(out, "one MoE layer:").split()
    return int(words[0]), int(words[3])


def test_moe(tmp_path):
    out = run(["moe"], cwd=tmp_path)
    assert line_value(out, "stacked experts = routing token by token:") == "True"
    assert counts(out) == (E * 3 * D * H + D * E, K * 3 * D * H + D * E)


def test_moe_shared_expert(tmp_path):
    out = run(["moe", "--n-shared", "1"], cwd=tmp_path)
    assert line_value(out, "stacked experts = routing token by token:") == "True"
    assert counts(out) == ((E + 1) * 3 * D * H + D * E, (K + 1) * 3 * D * H + D * E)
