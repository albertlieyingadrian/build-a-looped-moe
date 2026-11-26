"""Stage 10: bank. Pass 2's core layers read four messages each from the previous pass's core blocks."""

from conftest import line_value, run


def test_bank(tmp_path):
    out = run(["bank"], cwd=tmp_path)
    assert line_value(out, "message reads per forward at K=2:").startswith("4")
    assert line_value(out, "bank:") == "1 finished core blocks + 1 active block"
    assert line_value(out, "messages per core layer:").startswith("4,")
    assert line_value(out, "zero queries: every message is the bank's average:") == "True"
