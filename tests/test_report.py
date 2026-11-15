"""The CUDA side is tested on the GPU by src/test.cu (both methods against a float64 CPU
reference). These tests cover the report: rows where standard attention did not fit, and
a run on the saved results."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
import report  # noqa: E402

HEADER = "method,causal,heads,n,fwd_ms,bwd_ms,fwd_tflops,bwd_tflops,extra_mb,max_diff_o,max_diff_dq\n"


def test_rows_that_did_not_fit_are_reported_as_out_of_memory(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(HEADER
                 + "flash,0,16,256,1.0,2.0,4.0,5.0,1.1,0.001,0.002\n"
                 + "standard,0,16,256,,,,,6.3,,\n"
                 + "flash,1,16,256,1.0,2.0,4.0,5.0,1.1,0.001,0.002\n"
                 + "standard,1,16,256,,,,,6.3,,\n")
    rows = report.read(p)
    assert not report.ran(rows[1]) and report.ran(rows[0])
    text = "\n".join(report.summary(rows))
    assert "out of memory" in text
    # fwd+bwd TFLOPS is total work over total time: (4 * 1 + 5 * 2) / 3
    assert "| 4.7 |" in text


@pytest.mark.skipif(not (ROOT / "results" / "rtx2080ti" / "attention.csv").exists(), reason="no results")
def test_report_runs_on_the_saved_results(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(report, "SUMMARY", tmp_path / "summary.md")
    report.main(quiet=True)
    assert "## Causal" in (tmp_path / "summary.md").read_text()
    assert {p.name for p in (tmp_path / "figures").iterdir()} == {"time.png", "memory.png"}
