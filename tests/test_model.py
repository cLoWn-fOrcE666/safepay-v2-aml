"""Phase 7: fast, deterministic unit tests for Phase 3/4/5 artifacts.

Uses the *saved* model and report artifacts (no retraining, no CSV reload). Builds
one synthetic in-memory transaction row to exercise the TreeSHAP explainer and the
offline narrative generator. Completes in a couple of seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import explainer as explainer_mod
from src.explainer import AMLTreeExplainer, generate_investigator_report

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "xgb_aml.json"
METRICS_PATH = ROOT / "models" / "metrics.json"

LEAKY_BALANCE = ("oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest")


# --------------------------------------------------------------------------- #
# Artifact / schema integrity
# --------------------------------------------------------------------------- #
def test_artifacts_exist() -> None:
    """Model and metrics artifacts must be present (Phase 3 outputs)."""
    assert MODEL_PATH.is_file(), f"Missing model artifact: {MODEL_PATH}"
    assert METRICS_PATH.is_file(), f"Missing metrics artifact: {METRICS_PATH}"


def test_feature_names_have_no_leaky_balance_columns() -> None:
    """metrics.json feature_names must not contain any leaky balance column."""
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    feature_names = metrics.get("feature_names")
    assert isinstance(feature_names, list) and feature_names, (
        "metrics.json has no populated 'feature_names'."
    )

    leaked = set(feature_names) & set(LEAKY_BALANCE) | (
        set(feature_names) & set(metrics.get("leakage_blocklist", []))
    )
    assert not leaked, f"Leaky columns present in feature_names: {sorted(leaked)}"


# --------------------------------------------------------------------------- #
# Explainer behaviour (additivity, calibration, offline determinism)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def explainer() -> AMLTreeExplainer:
    """Load the saved explainer once for the module (fast: no retraining)."""
    return AMLTreeExplainer(model_path=MODEL_PATH, metrics_path=METRICS_PATH)


@pytest.fixture(scope="module")
def sample_row(explainer: AMLTreeExplainer) -> pd.DataFrame:
    """Build a single synthetic row over the model's exact feature schema."""
    record: dict[str, object] = {}
    for name in explainer.feature_names:
        if name in explainer.categorical_cols:
            # Use a category known to the training data (fallback to first / "TRANSFER").
            known = explainer.category_categories.get(name) or ["TRANSFER"]
            preferred = "TRANSFER" if "TRANSFER" in known else known[0]
            record[name] = preferred
        elif name == "step":
            record[name] = 400
        elif name == "amount":
            record[name] = 4_559_921.56
        elif name == "dest_in_degree":
            record[name] = 1.0
        elif name == "dest_pagerank":
            record[name] = 0.001
        else:
            record[name] = 0.0
    return pd.DataFrame([record])[explainer.feature_names]


def test_shap_log_odds_additivity(
    explainer: AMLTreeExplainer, sample_row: pd.DataFrame
) -> None:
    """Exact TreeSHAP: base_log_odds + sum(all_shap) == raw_log_odds (< 1e-4)."""
    payload = explainer.explain_transaction(sample_row)

    # Full SHAP matrix over ALL features (not just the reported top-k drivers).
    all_shap = explainer.explain_batch(sample_row)
    total_shap = float(np.sum(all_shap))

    base = float(payload["base_log_odds"])
    raw = float(payload["raw_log_odds"])

    residual = abs(base + total_shap - raw)
    assert residual < 1e-4, (
        f"TreeSHAP additivity violated: |{base:.6f} + {total_shap:.6f} - {raw:.6f}| "
        f"= {residual:.2e} >= 1e-4"
    )


def test_predicted_probability_is_calibrated(
    explainer: AMLTreeExplainer, sample_row: pd.DataFrame
) -> None:
    """Predicted probability must lie in [0, 1] and match sigmoid(raw log-odds)."""
    payload = explainer.explain_transaction(sample_row)
    prob = float(payload["predicted_probability"])
    raw = float(payload["raw_log_odds"])

    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"
    expected = 1.0 / (1.0 + np.exp(-raw))
    assert abs(prob - expected) < 1e-6, (
        f"Probability {prob} != sigmoid(raw) {expected}"
    )

    # is_flagged must be consistent with the threshold comparison.
    threshold = float(payload["threshold"])
    assert bool(payload["is_flagged"]) == bool(prob >= threshold)


def test_offline_report_is_deterministic(
    explainer: AMLTreeExplainer,
    sample_row: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no LLM key, generate_investigator_report is deterministic offline text."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    payload = explainer.explain_transaction(sample_row)

    report_a = generate_investigator_report(payload)
    report_b = generate_investigator_report(payload)

    assert isinstance(report_a, str) and report_a.strip()
    assert report_a == report_b, "Offline investigator report is not deterministic."
    # The deterministic NLG template cites drivers and is a 2-sentence note.
    assert "SHAP" in report_a or "drivers" in report_a
