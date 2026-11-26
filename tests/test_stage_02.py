"""Stage 02: blocks. Finished blocks plus the active block add up to the residual stream."""

import re

from conftest import line_value, run


def test_blocks(tmp_path):
    out = run(["blocks"], cwd=tmp_path)
    counts = [int(n) for n in re.findall(r"after layer \d+: (\d+) finished blocks", out)]
    assert counts == [1, 1, 2, 2]
    assert float(line_value(out, "finished blocks + active block = the residual stream, max difference:")) < 1e-4
