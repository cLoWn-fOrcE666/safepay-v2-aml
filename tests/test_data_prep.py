"""Phase 7: fast, deterministic unit tests for Phase 1 (data preparation).

Runs entirely on synthetic in-memory DataFrames — never touches the 6.35M-row
raw CSV — so the whole module completes in well under a second.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data_prep

# The full set of columns that must be dropped as leakage / post-hoc audit data.
# This mirrors the task's expectation and model.LEAKAGE_BLOCKLIST.
EXPECTED_LEAKY = (
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "isFlaggedFraud",
)


def _synthetic_raw(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Build a tiny PaySim-shaped frame including leaky and id columns.

    ``step`` is assigned a *unique* value per row (1..n) then row order is
    shuffled, so that an index-based chronological split always yields splits
    with strictly disjoint ``step`` ranges (zero overlap). This mirrors the real
    PaySim invariant where a ``step`` belongs to exactly one temporal bucket.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(1, n + 1)  # unique, dense, contiguous
    rng.shuffle(steps)
    df = pd.DataFrame(
        {
            "step": steps,
            "type": rng.choice(
                ["TRANSFER", "CASH_OUT", "PAYMENT"], size=n
            ),
            "amount": rng.uniform(1.0, 1e6, size=n).round(2),
            "nameOrig": [f"O{i}" for i in range(n)],
            "nameDest": [f"D{i}" for i in range(n)],
            # Leaky / post-event columns:
            "oldbalanceOrg": rng.uniform(0, 1e6, size=n).round(2),
            "newbalanceOrig": rng.uniform(0, 1e6, size=n).round(2),
            "oldbalanceDest": rng.uniform(0, 1e6, size=n).round(2),
            "newbalanceDest": rng.uniform(0, 1e6, size=n).round(2),
            "isFlaggedFraud": rng.integers(0, 2, size=n),
            "isFraud": rng.integers(0, 2, size=n),
        }
    )
    return df


def test_leaky_columns_constant_includes_all_five() -> None:
    """LEAKY_COLUMNS must enumerate all five leakage/audit columns."""
    declared = set(data_prep.LEAKY_COLUMNS)
    missing = set(EXPECTED_LEAKY) - declared
    assert not missing, f"LEAKY_COLUMNS missing required leaky columns: {missing}"


def test_drop_leaky_removes_balance_and_flag_columns() -> None:
    """`_drop_leaky` removes every leaky column and retains the rest."""
    df = _synthetic_raw()
    cleaned = data_prep._drop_leaky(df)

    for col in EXPECTED_LEAKY:
        assert col not in cleaned.columns, f"Leaky column '{col}' survived drop."

    # Non-leaky columns (incl. identifiers needed by Phase 2) are preserved.
    for col in ("step", "type", "amount", "nameOrig", "nameDest", "isFraud"):
        assert col in cleaned.columns, f"Expected column '{col}' was dropped."


def test_leaky_columns_absent_from_model_features() -> None:
    """No leaky/balance column may appear in the model's feature blocklist or
    the leakage-free feature lists built from a prepared schema."""
    from src import model

    # The model's blocklist must cover every leaky column.
    assert set(EXPECTED_LEAKY).issubset(model.LEAKAGE_BLOCKLIST), (
        "model.LEAKAGE_BLOCKLIST does not cover all leaky columns."
    )

    # A fully-featured prepared schema (post-drop-leaky has no balance columns,
    # but include them to prove _resolve_feature_lists never selects them).
    columns = list(
        dict.fromkeys(
            [
                "step",
                "type",
                "amount",
                *model.GRAPH_FEATURES,
                "isFraud",
                "nameOrig",
                "nameDest",
                *EXPECTED_LEAKY,
            ]
        )
    )
    tabular, all_features = model._resolve_feature_lists(columns)
    resolved = set(tabular) | set(all_features)
    assert not (resolved & set(EXPECTED_LEAKY)), (
        f"Leaky columns leaked into model features: {resolved & set(EXPECTED_LEAKY)}"
    )


def test_temporal_split_is_chronological_with_no_overlap() -> None:
    """train/val/test must be ordered in time with zero step overlap."""
    df = _synthetic_raw(n=400)
    train, val, test = data_prep._temporal_split(df)

    # Non-empty and sized by the configured fractions.
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    assert len(train) + len(val) + len(test) == len(df)

    # Core temporal invariant: no future data precedes earlier splits.
    assert train["step"].max() <= val["step"].min()
    assert val["step"].max() <= test["step"].min()

    # Zero step overlap across adjacent splits.
    assert set(train["step"]).isdisjoint(set(val["step"]))
    assert set(val["step"]).isdisjoint(set(test["step"]))
    assert set(train["step"]).isdisjoint(set(test["step"]))

    # Each split must be individually sorted by step ascending.
    assert train["step"].is_monotonic_increasing
    assert val["step"].is_monotonic_increasing
    assert test["step"].is_monotonic_increasing
