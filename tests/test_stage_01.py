"""Stage 01: recap. A plain looped GPT: K adds visits, not parameters, and training lowers the loss."""

import re

from conftest import line_value, run


def test_recap(tmp_path):
    out = run(["recap", "--steps", "150"], cwd=tmp_path)
    rows = {int(k): (int(p), int(v)) for k, p, v in re.findall(r"K=(\d+)\s+parameters (\d+)\s+layer visits (\d+)", out)}
    assert set(rows) == {1, 3, 6}
    assert len({p for p, _ in rows.values()}) == 1, "the parameter count must not depend on K"
    for k, (_, v) in rows.items():
        assert v == 2 + 4 * k + 2, "visits = prelude + K * core + coda layers (2 + 4K + 2 at the toy layout)"
    first = float(line_value(out, "loss over the first 20 steps:"))
    last = float(line_value(out, "loss over the last 20 steps:"))
    assert last < first
