"""Stage 08: balance. Nudging the router bias evens out the load."""

from conftest import line_value, run


def test_balance(tmp_path):
    out = run(["balance"], cwd=tmp_path)
    before = float(line_value(out, "load max/mean before balancing:"))
    after = float(line_value(out, "load max/mean after 300 balance steps:"))
    assert after < before and after < 1.4
