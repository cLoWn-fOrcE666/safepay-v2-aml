"""Phase 3: XGBoost fraud model, imbalance handling, and graph-vs-tabular ablation.

Trains a CPU-optimized gradient-boosted classifier on the Phase 2 graph-enriched
splits, tuned for Area Under the Precision-Recall Curve (AUPRC). Includes:

* cost-sensitive weighting (``scale_pos_weight``) with an unweighted ablation,
* validation-based F1-optimal threshold selection and Precision@80%-Recall,
* a locked evaluation on test (AUPRC, ROC-AUC, Precision, Recall, FPR, alerts),
* a tabular-only baseline (8 graph features dropped) on the identical split to
  quantify the AUPRC lift contributed by the Phase 2 graph features.

Artifacts: ``models/xgb_aml.json`` (trained model) and ``models/metrics.json``
(metrics + feature importances). No execution is performed at import time.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

PROCESSED_DIR: Final[Path] = Path("data/processed")
MODELS_DIR: Final[Path] = Path("models")

LABEL_COL: Final[str] = "isFraud"
CATEGORICAL_COLS: Final[tuple[str, ...]] = ("type",)

# The eight Phase 2 structural graph features.
GRAPH_FEATURES: Final[tuple[str, ...]] = (
    "orig_in_degree",
    "orig_out_degree",
    "orig_pagerank",
    "orig_clustering",
    "dest_in_degree",
    "dest_out_degree",
    "dest_pagerank",
    "dest_clustering",
)

# Leak-free pre-event tabular features available in the Phase 2 splits. Balance
# columns (old/new) and any post-hoc audit fields are DELIBERATELY excluded
# (see docs/ADR/001-temporal-split-and-leakage.md).
TABULAR_FEATURES: Final[tuple[str, ...]] = (
    "step",
    "type",
    "amount",
)

# Optional cyclical time features, included only when present in the parquet.
OPTIONAL_TIME_FEATURES: Final[tuple[str, ...]] = ("hour_of_day", "day_of_week")

# Columns that must NEVER be used as model features (leakage / identifiers).
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

# Target precision-recall operating point for the secondary threshold metric.
RECALL_TARGET: Final[float] = 0.80

# Fixed XGBoost hyper-parameters (CPU, hist method).
XGB_PARAMS: Final[dict[str, object]] = {
    "tree_method": "hist",
    "n_estimators": 600,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}
EARLY_STOPPING_ROUNDS: Final[int] = 50


@dataclass(frozen=True)
class SplitData:
    """Encoded features and label for a single split."""

    x: pd.DataFrame
    y: np.ndarray


@dataclass(frozen=True)
class ThresholdResult:
    """Operating point chosen on the validation set."""

    threshold: float
    f1: float
    precision_at_recall: float


def _load(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing '{path}'. Run Phase 2 (src/graph_engine.py) first."
        )
    return pd.read_parquet(path)


def _encode_categoricals(
    frames: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Encode ``type`` as an ordered category with a union of observed values."""
    categories: list[str] = sorted(
        {
            str(value)
            for frame in frames.values()
            for value in frame[CATEGORICAL_COLS[0]].unique()
        }
    )
    encoded: dict[str, pd.DataFrame] = {}
    for name, frame in frames.items():
        frame = frame.copy()
        for col in CATEGORICAL_COLS:
            frame[col] = pd.Categorical(frame[col].astype(str), categories=categories)
        encoded[name] = frame
    return encoded


def _resolve_feature_lists(columns: list[str]) -> tuple[list[str], list[str]]:
    """Build leakage-free ``(tabular_features, all_features)`` for the splits.

    ``TABULAR_FEATURES`` are the fixed pre-event columns; optional cyclical time
    features are appended only when actually present in the parquet schema.
    ``all_features`` = tabular + the eight graph features. A hard assertion
    guarantees no blocklisted (leaky / identifier) column slips into either list.
    """
    tabular: list[str] = [c for c in TABULAR_FEATURES if c in columns]
    tabular += [c for c in OPTIONAL_TIME_FEATURES if c in columns]
    all_features: list[str] = tabular + [c for c in GRAPH_FEATURES if c in columns]

    missing_core = [c for c in TABULAR_FEATURES if c not in columns]
    if missing_core:
        raise ValueError(f"Parquet is missing required tabular columns: {missing_core}")
    missing_graph = [c for c in GRAPH_FEATURES if c not in columns]
    if missing_graph:
        raise ValueError(f"Parquet is missing required graph columns: {missing_graph}")

    leaked_tabular = set(tabular) & LEAKAGE_BLOCKLIST
    leaked_all = set(all_features) & LEAKAGE_BLOCKLIST
    if leaked_tabular or leaked_all:
        raise AssertionError(
            f"Leakage blocklist violation: tabular={sorted(leaked_tabular)} "
            f"all={sorted(leaked_all)}"
        )
    return tabular, all_features


def _to_split(frame: pd.DataFrame, feature_cols: list[str]) -> SplitData:
    x = frame[feature_cols].copy()
    y = frame[LABEL_COL].to_numpy(dtype=np.int8)
    return SplitData(x=x, y=y)


def _build_classifier(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )


def _train(
    train: SplitData, val: SplitData, scale_pos_weight: float
) -> XGBClassifier:
    """Fit a classifier with early stopping on the validation AUPRC."""
    model = _build_classifier(scale_pos_weight)
    model.fit(
        train.x,
        train.y,
        eval_set=[(val.x, val.y)],
        verbose=False,
    )
    return model


def _predict_scores(model: XGBClassifier, x: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def _optimal_f1_threshold(y: np.ndarray, scores: np.ndarray) -> ThresholdResult:
    """Pick the probability threshold that maximizes F1 on validation."""
    precision, recall, thresholds = precision_recall_curve(y, scores)
    # precision/recall have length len(thresholds) + 1 (last point is 1/0).
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    best_idx = int(np.argmax(f1[:-1]))
    threshold = float(thresholds[best_idx])
    best_f1 = float(f1[best_idx])
    precision_at_recall = _precision_at_recall(precision, recall, RECALL_TARGET)
    return ThresholdResult(
        threshold=threshold,
        f1=best_f1,
        precision_at_recall=precision_at_recall,
    )


def _precision_at_recall(
    precision: np.ndarray, recall: np.ndarray, target_recall: float
) -> float:
    """Precision at the smallest threshold whose recall still meets the target."""
    mask = recall >= target_recall
    if not mask.any():
        return 0.0
    return float(precision[mask][np.argmax(precision[mask])])


def _evaluate_at_threshold(
    y: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """Compute the locked operating-point metrics for an evaluated split."""
    predictions = (scores >= threshold).astype(np.int8)
    tp = int(np.sum((predictions == 1) & (y == 1)))
    fp = int(np.sum((predictions == 1) & (y == 0)))
    fn = int(np.sum((predictions == 0) & (y == 1)))
    tn = int(np.sum((predictions == 0) & (y == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    alert_volume = int(tp + fp)

    return {
        "auprc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "precision": precision,
        "recall": recall,
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "fpr": fpr,
        "alert_volume": alert_volume,
    }


def _feature_importances(model: XGBClassifier, feature_cols: list[str]) -> dict[str, float]:
    gains = model.feature_importances_
    return {name: float(score) for name, score in zip(feature_cols, gains)}


def _run_experiment(
    name: str,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    scale_pos_weight: float,
) -> dict[str, object]:
    """Train one model, tune the threshold on val, evaluate on test."""
    model = _train(train, val, scale_pos_weight)
    val_scores = _predict_scores(model, val.x)
    threshold = _optimal_f1_threshold(val.y, val_scores)

    test_metrics = _evaluate_at_threshold(
        test.y, _predict_scores(model, test.x), threshold.threshold
    )
    logger.info("[%s] val F1-opt thr=%.4f -> test AUPRC=%.4f", name, threshold.threshold, test_metrics["auprc"])

    return {
        "model": model,
        "threshold": threshold.threshold,
        "val_f1": threshold.f1,
        "val_precision_at_80_recall": threshold.precision_at_recall,
        "test_metrics": test_metrics,
        "best_iteration": int(model.best_iteration),
    }


def _jsonable(value: object) -> object:
    """Recursively convert numpy scalar/array types to JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def run(processed_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR) -> None:
    """Execute the full Phase 3 pipeline and persist model + metrics artifacts."""
    models_dir.mkdir(parents=True, exist_ok=True)

    raw = {
        "train": _load(processed_dir / "train_graph.parquet"),
        "val": _load(processed_dir / "val_graph.parquet"),
        "test": _load(processed_dir / "test_graph.parquet"),
    }
    encoded = _encode_categoricals(raw)

    tabular_features, all_features = _resolve_feature_lists(
        list(encoded["train"].columns)
    )
    logger.info(
        "Leakage-free features: %d tabular + %d graph = %d total",
        len(tabular_features),
        len(all_features) - len(tabular_features),
        len(all_features),
    )

    # --- Primary model: full feature set (graph + tabular), weighted. -------
    train_full = _to_split(encoded["train"], all_features)
    val_full = _to_split(encoded["val"], all_features)
    test_full = _to_split(encoded["test"], all_features)

    pos = int(train_full.y.sum())
    neg = int(len(train_full.y) - pos)
    scale_pos_weight = neg / pos if pos else 1.0

    weighted = _run_experiment(
        "full_weighted", train_full, val_full, test_full, scale_pos_weight
    )
    # Unweighted ablation (scale_pos_weight = 1.0) for imbalance comparison.
    unweighted = _run_experiment(
        "full_unweighted", train_full, val_full, test_full, 1.0
    )

    # --- Tabular-only baseline: identical split, 8 graph features dropped. --
    train_tab = _to_split(encoded["train"], tabular_features)
    val_tab = _to_split(encoded["val"], tabular_features)
    test_tab = _to_split(encoded["test"], tabular_features)
    tabular_only = _run_experiment(
        "tabular_only", train_tab, val_tab, test_tab, scale_pos_weight
    )

    # --- Persist the locked weighted model + feature importances. -----------
    final_model: XGBClassifier = weighted["model"]
    final_model.save_model(models_dir / "xgb_aml.json")

    graph_auprc = float(weighted["test_metrics"]["auprc"])
    tab_auprc = float(tabular_only["test_metrics"]["auprc"])

    metrics = {
        "label": LABEL_COL,
        "class_balance": {
            "train_positive": pos,
            "train_negative": neg,
            "scale_pos_weight": scale_pos_weight,
        },
        "selected_model": "full_weighted",
        "threshold": weighted["threshold"],
        "val": {
            "f1_optimal": weighted["val_f1"],
            "precision_at_80_recall": weighted["val_precision_at_80_recall"],
        },
        "test": weighted["test_metrics"],
        "ablation": {
            "full_weighted": {
                "test": weighted["test_metrics"],
                "threshold": weighted["threshold"],
                "best_iteration": weighted["best_iteration"],
            },
            "full_unweighted": {
                "test": unweighted["test_metrics"],
                "threshold": unweighted["threshold"],
                "best_iteration": unweighted["best_iteration"],
            },
            "tabular_only": {
                "test": tabular_only["test_metrics"],
                "threshold": tabular_only["threshold"],
                "best_iteration": tabular_only["best_iteration"],
            },
            "graph_feature_auprc_lift": graph_auprc - tab_auprc,
        },
        "feature_importances": _feature_importances(final_model, all_features),
        "feature_names": list(all_features),
        "graph_features": list(GRAPH_FEATURES),
        "tabular_features": tabular_features,
        "leakage_blocklist": sorted(LEAKAGE_BLOCKLIST),
    }

    with (models_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(metrics), handle, indent=2)

    logger.info(
        "Graph AUPRC lift on test: %.4f (full=%.4f vs tabular=%.4f)",
        graph_auprc - tab_auprc,
        graph_auprc,
        tab_auprc,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
