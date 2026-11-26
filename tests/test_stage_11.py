"""Stage 11: gate. A new message gate is closed, so K changes nothing until the gates open."""

from conftest import line_value, run


def test_gate(tmp_path):
    out = run(["gate"], cwd=tmp_path)
    assert line_value(out, "a new gate passes x through unchanged:") == "True"
    assert float(line_value(out, "new model: change in logits from K=1 to K=3:")) < 1e-5
    assert float(line_value(out, "gates opened: change in logits from K=1 to K=3:")) > 1e-3
