from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.reporting.supplement_tables import (
    f3, f4, fint, render_simple_tabular,
)


def test_formatting_rules():
    assert f3(0.7436000001) == "0.744"
    assert f4(0.0763574214) == "0.0764"
    assert f4(-0.0624910126) == "-0.0625"
    assert fint(5143.7666) == "5,144"


def test_tex_renderer_is_body_only():
    x = render_simple_tabular(["A","B"], [["x","1"]])
    assert "\\begin{table" not in x
    assert "\\caption" not in x
    assert "\\label" not in x
    assert "\\begin{tabular}" in x
    assert "\\toprule" in x
    assert "\\bottomrule" in x


def test_generator_has_no_scientific_result_literals():
    text = (
        ROOT / "experiments" / "reporting" / "generate_supplement_tables.py"
    ).read_text(encoding="utf-8")
    for forbidden in [
        "0.7436000000000001",
        "0.9090666666666667",
        "0.8737004111111111",
        "0.9973519971116672",
        "0.7794222222222221",
    ]:
        assert forbidden not in text


def test_generator_does_not_read_gold():
    text = (
        ROOT / "experiments" / "reporting" / "generate_supplement_tables.py"
    ).read_text(encoding="utf-8")
    assert "paper_results/expected" not in text
    assert "supplement_table_gold" not in text
