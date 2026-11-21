"""Phase 4–5: TreeSHAP explainability + LLM investigator reports (SafePay 2.0).

Phase 4 computes exact TreeSHAP attributions using XGBoost's native C++ engine
(``Booster.predict(..., pred_contribs=True)``), which implements Lundberg's exact
path-dependent TreeSHAP algorithm ($O(TLD^2)$) directly in raw log-odds space.

Phase 5 adds :func:`generate_investigator_report`, a provider-agnostic AML
narrative generator that turns a per-alert SHAP payload into a concise 2-sentence
Suspicious Activity Report triage note. It calls any OpenAI-compatible chat
endpoint when an API key is configured, and otherwise (or on any API error) falls
back to a clean, deterministic NLG template.

Local accuracy therefore holds exactly::

    raw_margin(x) = base_log_odds + sum_i shap_i(x)

Using the native booster avoids the ``base_score`` intercept parsing mismatch
between ``shap.TreeExplainer`` and XGBoost's JSON model header on newer Python
versions, so the intercept and per-feature contributions come from the same C++
kernel that produced the model.

The class is designed to be imported by Phase 5 (LLM investigator synthesis) and
Phase 6 (Streamlit dashboard). Its public methods return JSON-serializable
payloads, so downstream layers never touch numpy/booster internals.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

MODELS_DIR: Final[Path] = Path("models")
REPORTS_DIR: Final[Path] = Path("reports")
PROCESSED_DIR: Final[Path] = Path("data/processed")

MODEL_PATH: Final[Path] = MODELS_DIR / "xgb_aml.json"
METRICS_PATH: Final[Path] = MODELS_DIR / "metrics.json"

LABEL_COL: Final[str] = "isFraud"
CATEGORICAL_COLS: Final[tuple[str, ...]] = ("type",)

# Columns that must NEVER appear as model features (leakage / identifiers).
# Mirrors src.model.LEAKAGE_BLOCKLIST; single source of truth is metrics.json.
LEAKAGE_BLOCKLIST: Final[frozenset[str]] = frozenset(
    {
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
        "isFraud",
        "nameOrig",
        "nameDest",
    }
)

# Tolerance for the TreeSHAP local-accuracy invariant. Matches XGBoost's internal
# float32 accumulation precision between pred_contribs and output_margin.
LOCAL_ACCURACY_TOL: Final[float] = 1e-4

# CLI sampling configuration.
N_SAMPLE: Final[int] = 2_000
MAX_FRAUD_SAMPLE: Final[int] = 500
TOP_ALERTS: Final[int] = 25
DEFAULT_TOP_K: Final[int] = 3


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic transform of raw log-odds to probability."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


class AMLTreeExplainer:
    """Exact TreeSHAP explainer for the trained SafePay AML XGBoost model."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metrics_path: Path = METRICS_PATH,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Missing model '{model_path}'. Run Phase 3 (src/model.py) first."
            )
        if not metrics_path.is_file():
            raise FileNotFoundError(
                f"Missing metrics '{metrics_path}'. Run Phase 3 (src/model.py) first."
            )

        self.model: XGBClassifier = XGBClassifier()
        self.model.load_model(model_path)

        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics: dict[str, object] = json.load(handle)
        self.threshold: float = float(metrics["threshold"])

        # Reproduce Phase 3's exact categorical encoding. XGBoost learns splits
        # keyed by category *code*, so the ``type`` category ordering must match
        # training (sorted union across all splits) or codes would misalign.
        self.category_categories: dict[str, list[str]] = {
            col: self._training_categories(col) for col in CATEGORICAL_COLS
        }

        # Authoritative feature ordering: prefer ``metrics.json["feature_names"]``
        # (the single source of truth written by Phase 3) and cross-check it
        # against the booster's own feature_names so SHAP attributions line up.
        self.booster = self.model.get_booster()
        booster_names = self.booster.feature_names
        if not booster_names:
            raise ValueError("Loaded booster has no feature_names.")
        metrics_names = metrics.get("feature_names")
        if isinstance(metrics_names, list) and metrics_names:
            if list(metrics_names) != list(booster_names):
                raise ValueError(
                    "metrics.json feature_names disagree with booster.feature_names: "
                    f"{list(metrics_names)} != {list(booster_names)}"
                )
            self.feature_names: list[str] = list(metrics_names)
        else:
            logger.warning(
                "metrics.json has no 'feature_names'; falling back to booster order."
            )
            self.feature_names = list(booster_names)

        # Fail loudly if any leaky/identifier column leaked into the feature set.
        leaked = set(self.feature_names) & LEAKAGE_BLOCKLIST
        if leaked:
            raise AssertionError(
                f"Leakage blocklist violation in explainer feature_names: {sorted(leaked)}"
            )

        # Authoritative categorical-vs-numeric split from the booster metadata
        # (``feature_types`` is "c" for categorical, otherwise numeric).
        feature_types = self.booster.feature_types or ["float"] * len(self.feature_names)
        self.categorical_cols: set[str] = {
            name
            for name, ftype in zip(self.feature_names, feature_types)
            if ftype == "c"
        }
        self.numeric_cols: list[str] = [
            name for name in self.feature_names if name not in self.categorical_cols
        ]

        # Base log-odds (intercept) is populated lazily from the final column of
        # the ``pred_contribs`` output the first time attributions are computed,
        # so it always comes from the same C++ TreeSHAP kernel as the per-feature
        # contributions (no fragile ``base_score`` string parsing).
        self.base_value: float | None = None

    @staticmethod
    def _training_categories(column: str) -> list[str]:
        """Rebuild the sorted-union category order used by Phase 3 encoding.

        Phase 3 encoded categoricals with a sorted union of observed values
        across the train/val/test splits. The training split alone contains the
        complete PaySim ``type`` domain (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER),
        so deriving the sorted set from ``train_graph.parquet`` reproduces the
        exact category codes the booster expects.
        """
        train_path = PROCESSED_DIR / "train_graph.parquet"
        if train_path.is_file():
            series = pd.read_parquet(train_path, columns=[column])[column]
            return sorted({str(v) for v in series.astype(str).unique()})
        logger.warning(
            "Training split not found; deriving '%s' categories from defaults.", column
        )
        return []

    # ------------------------------------------------------------------ #
    # Input preparation
    # ------------------------------------------------------------------ #
    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Align and coerce a frame to the model's exact feature schema.

        Mixed-dtype frame extraction (e.g. a ``pd.Series`` row via
        ``row.to_frame().T``) upcasts numeric columns to ``object``. We therefore
        explicitly restore dtypes: categorical features become ``pd.Categorical``
        (using training categories) and numeric features are coerced back to
        numeric, so ``xgb.DMatrix`` always receives int/float/bool/category.
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing model features: {missing}")
        aligned = X.loc[:, self.feature_names].copy()

        # Categorical columns -> pd.Categorical with training category ordering.
        for col in self.categorical_cols:
            if col not in aligned.columns:
                continue
            raw = aligned[col]
            # A categorical Series already carries the right categories; only
            # rebuild from strings when it was upcast to ``object``.
            if not isinstance(raw.dtype, pd.CategoricalDtype):
                values = raw.astype(str)
                categories = self.category_categories.get(col) or sorted(
                    {str(v) for v in values.unique()}
                )
                aligned[col] = pd.Categorical(values, categories=categories)
                raw = aligned[col]
            if raw.isna().any():
                known = set(raw.cat.categories)
                unknown = sorted({str(v) for v in raw.dropna().unique()} - known)
                if unknown:
                    raise ValueError(
                        f"Unknown '{col}' categories not seen in training: {unknown}"
                    )

        # Numeric columns -> strict numeric coercion (raises on non-numeric).
        for col in self.numeric_cols:
            if col in aligned.columns:
                aligned[col] = pd.to_numeric(aligned[col], errors="raise")

        return aligned

    def _build_dmatrix(self, X_prepared: pd.DataFrame) -> xgb.DMatrix:
        """Build a DMatrix preserving exact feature names/types the model expects."""
        return xgb.DMatrix(
            X_prepared,
            feature_names=self.feature_names,
            enable_categorical=True,
        )

    def _raw_margin(self, X_prepared: pd.DataFrame) -> np.ndarray:
        """Raw (pre-sigmoid) log-odds margin via ``output_margin=True``."""
        dmatrix = self._build_dmatrix(X_prepared)
        return self.booster.predict(dmatrix, output_margin=True).astype(np.float64)

    # ------------------------------------------------------------------ #
    # SHAP computation (native C++ exact TreeSHAP)
    # ------------------------------------------------------------------ #
    def _contribs(
        self, X_prepared: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact TreeSHAP ``(shap_values [n, f], base_values [n])``.

        Uses XGBoost's native ``pred_contribs=True`` (Lundberg's exact
        path-dependent TreeSHAP, $O(TLD^2)$), which returns an
        ``(n_samples, n_features + 1)`` array. The trailing column is the exact
        log-odds intercept ``phi_0`` for each row.
        """
        dmatrix = self._build_dmatrix(X_prepared)
        contribs = self.booster.predict(dmatrix, pred_contribs=True)
        contribs = np.asarray(contribs, dtype=np.float64)
        if contribs.ndim == 1:
            contribs = contribs.reshape(1, -1)
        shap_values = contribs[:, :-1]
        base_values = contribs[:, -1]
        # Cache the exact intercept from the final column of ``pred_contribs``.
        self.base_value = float(contribs[0, -1])
        return shap_values, base_values

    def explain_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Return the ``[n_samples, n_features]`` SHAP matrix in log-odds space."""
        prepared = self._prepare(X)
        values, base_values = self._contribs(prepared)
        self._verify_local_accuracy(prepared, values, base_values)
        return values

    def _verify_local_accuracy(
        self,
        prepared: pd.DataFrame,
        values: np.ndarray,
        base_values: np.ndarray,
    ) -> None:
        """Assert exact additivity: ``base + sum(phi) == raw_margin`` per row."""
        margin = self._raw_margin(prepared)
        reconstructed = base_values + values.sum(axis=1)
        max_err = float(np.max(np.abs(reconstructed - margin))) if len(margin) else 0.0
        if max_err >= LOCAL_ACCURACY_TOL:
            raise AssertionError(
                f"TreeSHAP local accuracy violated: max error {max_err:.6e} "
                f">= {LOCAL_ACCURACY_TOL}"
            )

    # ------------------------------------------------------------------ #
    # Per-transaction explanation
    # ------------------------------------------------------------------ #
    def explain_transaction(
        self, row: pd.Series | pd.DataFrame, top_k: int = DEFAULT_TOP_K
    ) -> dict[str, object]:
        """Explain a single transaction and return a JSON-serializable payload."""
        if isinstance(row, pd.Series):
            frame = row.to_frame().T
        else:
            frame = row.iloc[[0]]
        prepared = self._prepare(frame)
        values, base_values = self._contribs(prepared)
        self._verify_local_accuracy(prepared, values, base_values)

        phi = values[0]
        base_value = float(base_values[0])
        raw_margin = float(self._raw_margin(prepared)[0])
        probability = float(_sigmoid(np.asarray([raw_margin]))[0])

        order = np.argsort(np.abs(phi))[::-1][:top_k]
        top_drivers: list[dict[str, object]] = []
        for idx in order:
            feature = self.feature_names[int(idx)]
            contribution = float(phi[int(idx)])
            top_drivers.append(
                {
                    "feature": feature,
                    "value": self._scalar(prepared.iloc[0][feature]),
                    "shap_log_odds": contribution,
                    "abs_impact": abs(contribution),
                    "direction": (
                        "INCREASES_RISK" if contribution > 0 else "DECREASES_RISK"
                    ),
                }
            )

        return {
            "predicted_probability": probability,
            "raw_log_odds": raw_margin,
            "base_log_odds": base_value,
            "threshold": self.threshold,
            "is_flagged": bool(probability >= self.threshold),
            "top_drivers": top_drivers,
        }

    # ------------------------------------------------------------------ #
    # Global importance
    # ------------------------------------------------------------------ #
    def compute_global_importance(self, X_sample: pd.DataFrame) -> list[dict[str, object]]:
        """Mean absolute SHAP per feature (log-odds), sorted descending."""
        values = self.explain_batch(X_sample)
        mean_abs = np.abs(values).mean(axis=0)
        ranking = np.argsort(mean_abs)[::-1]
        return [
            {
                "feature": self.feature_names[int(idx)],
                "mean_abs_shap_log_odds": float(mean_abs[int(idx)]),
            }
            for idx in ranking
        ]

    @staticmethod
    def _scalar(value: object) -> object:
        """Convert a numpy/pandas scalar to a native JSON-friendly type."""
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value if isinstance(value, (str, float, int, bool)) else str(value)


# ---------------------------------------------------------------------- #
# Phase 5: Provider-agnostic AML narrative generation
# ---------------------------------------------------------------------- #
INVESTIGATOR_SYSTEM_PROMPT: Final[str] = (
    "You are a senior AML compliance analyst. Write a concise 2-sentence "
    "Suspicious Activity Report triage note citing ONLY the exact feature values "
    "and log-odds SHAP drivers provided. Do not invent numbers."
)
DEFAULT_LLM_BASE_URL: Final[str] = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL: Final[str] = "gpt-4o-mini"


def _format_value(value: object) -> str:
    """Render a feature value compactly and deterministically for narrative text."""
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return str(value)


def _render_deterministic_report(explanation: dict) -> str:
    """Deterministic NLG template summarising a SHAP payload as a SAR triage note.

    Converts the Top-3 TreeSHAP drivers, their exact feature values, log-odds
    shifts, and the calibrated risk probability into a crisp two-sentence note.
    Runs fully offline; never calls an external service.
    """
    prob = explanation.get("predicted_probability", 0.0)
    threshold = explanation.get("threshold", 0.0)
    flagged = bool(explanation.get("is_flagged", False))

    try:
        prob_str = f"{float(prob):.2%}"
    except (TypeError, ValueError):
        prob_str = str(prob)
    try:
        threshold_str = f"{float(threshold):.2%}"
    except (TypeError, ValueError):
        threshold_str = str(threshold)

    verdict = "crossed the alert threshold" if flagged else "stayed below the alert threshold"

    drivers = explanation.get("top_drivers") or []
    driver_clauses: list[str] = []
    for driver in drivers[:3]:
        feature = driver.get("feature", "unknown feature")
        value = _format_value(driver.get("value"))
        shap = driver.get("shap_log_odds", 0.0)
        try:
            shap_str = f"{float(shap):+.4f}"
        except (TypeError, ValueError):
            shap_str = str(shap)
        effect = "raised risk" if driver.get("direction") == "INCREASES_RISK" else "lowered risk"
        driver_clauses.append(f"{feature}={value} ({shap_str} log-odds, {effect})")

    if driver_clauses:
        drivers_text = ", ".join(driver_clauses[:-1]) + (
            f" and {driver_clauses[-1]}" if len(driver_clauses) > 1 else driver_clauses[0]
        )
    else:
        drivers_text = "no material SHAP drivers"

    sentence_1 = (
        f"Transaction flagged for review: the model's calibrated fraud probability "
        f"was {prob_str}, which {verdict} of {threshold_str}."
    )
    sentence_2 = f"The dominant TreeSHAP drivers were {drivers_text}."
    return f"{sentence_1} {sentence_2}"


def generate_investigator_report(
    explanation: dict,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> str:
    """Generate a 2-sentence AML investigator triage note for one alert.

    Provider-agnostic: any OpenAI-compatible chat endpoint can be used via the
    ``LLM_API_KEY`` / ``LLM_BASE_URL`` / ``LLM_MODEL`` environment variables or
    the explicit arguments. When no API key is configured (offline mode) — or the
    ``openai`` package is unavailable, or any API error occurs — it returns the
    deterministic NLG report from :func:`_render_deterministic_report`, so the
    pipeline always produces a well-formed triage note.
    """
    resolved_key = api_key or os.getenv("LLM_API_KEY")
    resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL
    resolved_model = model or os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL

    if not resolved_key:
        logger.info(
            "No LLM_API_KEY configured; using deterministic offline NLG report."
        )
        return _render_deterministic_report(explanation)

    user_payload = {
        "predicted_probability": explanation.get("predicted_probability"),
        "threshold": explanation.get("threshold"),
        "is_flagged": explanation.get("is_flagged"),
        "top_drivers": explanation.get("top_drivers", []),
    }

    try:
        from openai import OpenAI  # optional dependency; only needed online

        client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)
        response = client.chat.completions.create(
            model=resolved_model,
            temperature=0.0,
            max_tokens=120,
            timeout=15.0,
            messages=[
                {"role": "system", "content": INVESTIGATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
        )
        content = response.choices[0].message.content
        if not content or not content.strip():
            logger.warning("LLM returned empty content; using deterministic NLG report.")
            return _render_deterministic_report(explanation)
        return content.strip()
    except Exception as exc:  # noqa: BLE001 - any API/runtime error -> offline NLG
        logger.warning("LLM narrative generation failed (%s); using offline NLG.", exc)
        return _render_deterministic_report(explanation)


# ---------------------------------------------------------------------- #
# CLI entrypoint
# ---------------------------------------------------------------------- #
def _load_test_frame(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing '{path}'. Run Phase 2 (src/graph_engine.py) first."
        )
    return pd.read_parquet(path)


def _stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Sample up to 500 frauds + random legit rows to ~N_SAMPLE total."""
    if LABEL_COL in df.columns:
        fraud = df[df[LABEL_COL] == 1]
        legit = df[df[LABEL_COL] == 0]
    else:
        fraud = df.iloc[0:0]
        legit = df

    n_fraud = min(len(fraud), MAX_FRAUD_SAMPLE)
    fraud_sample = fraud.sample(n=n_fraud, random_state=1) if n_fraud else fraud

    n_legit = max(N_SAMPLE - n_fraud, 0)
    n_legit = min(n_legit, len(legit))
    legit_sample = legit.sample(n=n_legit, random_state=2) if n_legit else legit

    return pd.concat([fraud_sample, legit_sample], axis=0)


def _top_risk_alerts(explainer: AMLTreeExplainer, df: pd.DataFrame) -> pd.DataFrame:
    """Return rows sorted by descending predicted probability, top TOP_ALERTS."""
    numeric = df.drop(columns=[c for c in (LABEL_COL,) if c in df.columns])
    prepared = explainer._prepare(numeric)  # noqa: SLF001 - internal use in CLI
    margins = explainer._raw_margin(prepared)  # noqa: SLF001
    probs = _sigmoid(margins)

    scored = df.drop(columns=[c for c in (LABEL_COL,) if c in df.columns]).copy()
    scored["_probability"] = probs
    flagged = scored[scored["_probability"] >= explainer.threshold]
    candidates = flagged if not flagged.empty else scored
    return candidates.sort_values("_probability", ascending=False).head(TOP_ALERTS)


def _print_summary(
    global_importance: list[dict[str, object]],
    sample_explanation: dict[str, object] | None,
) -> None:
    print("\n=== Global Top-5 Features (mean |SHAP| in log-odds) ===")
    for rank, entry in enumerate(global_importance[:5], start=1):
        print(f"{rank}. {entry['feature']:<18} {entry['mean_abs_shap_log_odds']:.6f}")

    if sample_explanation is None:
        print("\n(No flagged alerts available for a sample explanation.)")
        return

    print("\n=== Sample Flagged Alert ===")
    print(
        f"predicted_probability={sample_explanation['predicted_probability']:.6f} "
        f"raw_log_odds={sample_explanation['raw_log_odds']:.4f} "
        f"base_log_odds={sample_explanation['base_log_odds']:.4f} "
        f"threshold={sample_explanation['threshold']:.6f} "
        f"flagged={sample_explanation['is_flagged']}"
    )
    print("Top-3 drivers:")
    for driver in sample_explanation["top_drivers"]:  # type: ignore[union-attr]
        print(
            f"  - {driver['feature']:<18} value={driver['value']!s:<12} "
            f"shap={driver['shap_log_odds']:+.4f} ({driver['direction']})"
        )

    report = sample_explanation.get("investigator_report")
    if report:
        print("\n=== Investigator Report ===")
        print(report)


def main() -> None:
    """Generate global + per-alert SHAP reports for the test split."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    explainer = AMLTreeExplainer()
    test = _load_test_frame(PROCESSED_DIR / "test_graph.parquet")

    feature_cols = explainer.feature_names
    sample = _stratified_sample(test)

    global_importance = explainer.compute_global_importance(sample[feature_cols])
    with (REPORTS_DIR / "global_shap_importance.json").open("w", encoding="utf-8") as handle:
        json.dump(global_importance, handle, indent=2)
    logger.info("Wrote %s", REPORTS_DIR / "global_shap_importance.json")

    top_alerts = _top_risk_alerts(explainer, test)
    explanations: list[dict[str, object]] = []
    for position in range(len(top_alerts)):
        # Slice a single-row *DataFrame* (double brackets) so original dtypes are
        # preserved — iterrows()/Series would upcast numerics to ``object``.
        row_df = top_alerts.iloc[[position]][feature_cols]
        payload = explainer.explain_transaction(row_df)
        payload["probability"] = float(top_alerts.iloc[position]["_probability"])
        payload["row_index"] = int(position)
        payload["investigator_report"] = generate_investigator_report(payload)
        explanations.append(payload)
    with (REPORTS_DIR / "sample_explanations.json").open("w", encoding="utf-8") as handle:
        json.dump(explanations, handle, indent=2)
    logger.info("Wrote %s", REPORTS_DIR / "sample_explanations.json")

    first_flag = next((e for e in explanations if e["is_flagged"]), None)
    _print_summary(global_importance, first_flag)


if __name__ == "__main__":
    main()
