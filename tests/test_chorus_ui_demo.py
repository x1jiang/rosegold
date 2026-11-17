"""Synthetic tests for app.chorus_ui_demo and Streamlit tab wiring in app.ui."""

from __future__ import annotations

import os
import py_compile
from pathlib import Path

from app.chorus_ui_demo import run_demo_chorus_sepsis_pipeline


def test_run_demo_chorus_sepsis_pipeline_without_env_mutation(monkeypatch):
    monkeypatch.setenv("ROSEGOLD_LLM_BACKEND", "llamacpp")
    monkeypatch.setenv("ROSEGOLD_ALLOW_MOCK", "0")

    res = run_demo_chorus_sepsis_pipeline()

    assert os.environ["ROSEGOLD_LLM_BACKEND"] == "llamacpp"
    assert os.environ["ROSEGOLD_ALLOW_MOCK"] == "0"

    assert res["manifest"]["encounters_processed"] == 6
    assert len(res["redcap_rows"]) == 6
    assert "sepsis_ync_v3" in res["redcap_csv"]
    assert "field_name" in res["evidence_csv"]
    assert "Stage-1 Three-Way Gate" in res["concordance_md"]
    assert res["concordance"]["gate_3way"]["overall_agreement_pct"] == 100.0
    assert res["concordance"]["gate_resolved"]["cohens_kappa"] == 1.0
    assert any(f["issue_type"] == "choice_code_mismatch" for f in res["audit_findings"])


def test_ui_compiles_and_exposes_redcap_tab():
    ui_path = Path(__file__).resolve().parent.parent / "app" / "ui.py"
    py_compile.compile(str(ui_path), doraise=True)
    text = ui_path.read_text(encoding="utf-8")
    assert "📋 REDCap Survey & Concordance" in text
    assert "render_chorus_redcap_tab()" in text
