"""Stage 03: mix. Depth attention: zero query = average, a pointed query picks a block, scale does not matter."""

from conftest import line_value, run


def test_mix(tmp_path):
    out = run(["mix"], cwd=tmp_path)
    assert [float(v) for v in line_value(out, "weights with a zero query:").split()] == [0.25] * 4
    pointed = [float(v) for v in line_value(out, "weights with the query pointed at block 1:").split()]
    assert pointed[1] >= 0.99
    assert float(line_value(out, "output change after scaling block 0 by 100:")) < 1e-5
