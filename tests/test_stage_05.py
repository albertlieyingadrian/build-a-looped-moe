"""Stage 05: final-mix. The head's own mix over 4 finished blocks + the active one."""

from conftest import line_value, run


def test_final_mix(tmp_path):
    out = run(["final-mix"], cwd=tmp_path)
    assert "the head mixes 4 finished blocks + the active one" in out
    assert [float(v) for v in line_value(out, "final_input weights at initialisation:").split()] == [0.2] * 5
    assert line_value(out, "logits = head(norm(final_input mix)):") == "True"
