from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_phase4_keeps_six_separate_electronics_figures():
    make = (ROOT / "Makefile").read_text(encoding="utf-8")
    win = (ROOT / "tools" / "run_phase4_windows.py").read_text(encoding="utf-8")
    for text in (make, win):
        assert "make_fig2_amazon_main.py" in text
        assert "make_fig3_amazon_robustness.py" in text
        assert "--stem-prefix" in text
        assert "electronics" in text
        assert "make_supp_electronics_replication.py" not in text


def test_windows_orchestrator_has_fresh_dual_category_raw_reproduction():
    text = (ROOT / "tools" / "run_phase4_windows.py").read_text(encoding="utf-8")
    for token in [
        "verify_raw()",
        "home_preprocess()",
        "home_downstream()",
        "electronics_preprocess()",
        "electronics_downstream()",
        "amazon_both_from_raw()",
        "reproduce_final_from_raw()",
    ]:
        assert token in text


def test_windows_reporting_includes_cross_category_table_and_electronics_reporting():
    text = (ROOT / "tools" / "run_phase4_windows.py").read_text(encoding="utf-8")
    assert "cross_category_table()" in text
    assert "electronics_reporting()" in text
    assert "home_figure_data()" in text
    assert '"reporting": reporting' in text


def test_windows_entry_point_requires_no_make():
    cmd = (ROOT / "RUN_PHASE4_WINDOWS.cmd").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "tools\\run_phase4_windows.py" in cmd
    assert "Windows reproduction does not require `make`" in readme
    assert "RUN_PHASE4_WINDOWS.cmd reproduce-final-from-raw" in readme
