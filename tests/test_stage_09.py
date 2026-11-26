"""Stage 09: loop. Every pass restarts from the prelude, so without messages K changes nothing."""

import re

from conftest import run


def test_loop(tmp_path):
    out = run(["loop"], cwd=tmp_path)
    changes = [float(v) for v in re.findall(r"change in logits from K=1 without messages: ([\d.e-]+)", out)]
    assert len(changes) == 3 and max(changes) < 1e-5
    assert "24 layers, 48 layer visits at K=3" in out
    assert "visit 18 = layer 6, pass 2" in out
