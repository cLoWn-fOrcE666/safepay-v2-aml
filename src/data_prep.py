"""Phase 1: Data preparation for SafePay 2.0.

Loads raw PaySim transactions, removes leaky columns, and performs a strict
temporal split by ``step`` (time bucket). Outputs ``train.parquet``,
``val.parquet``, and ``test.parquet`` to ``data/processed/`` preserving the raw
``nameOrig``/``nameDest`` string IDs needed by Phase 2.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_PATH: Final[Path] = Path("data/raw/paysim.csv")
PROCESSED_DIR: Final[Path] = Path("data/processed")

# Columns that leak the label or post-hoc audit information.
LEAKY_COLUMNS: Final[tuple[str, ...]] = ("isFlaggedFraud",)

# Temporal split fractions (chronological by ``step``).
TRAIN_FRAC: Final[float] = 0.70
VAL_FRAC: Final[float] = 0.15  # test gets the remainder


def _load_raw(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing raw dataset: {path}")
    return pd.read_csv(path)


def _drop_leaky(df: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in LEAKY_COLUMNS if c in df.columns]
    return df.drop(columns=drop)


def _temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split by ``step`` so no future data precedes train."""
    df = df.sort_values("step", kind="mergesort").reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    return train, val, test


def run(raw_path: Path = RAW_PATH, processed_dir: Path = PROCESSED_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    df = _load_raw(raw_path)
    df = _drop_leaky(df)

    train, val, test = _temporal_split(df)
    for name, frame in (("train", train), ("val", val), ("test", test)):
        out = processed_dir / f"{name}.parquet"
        frame.to_parquet(out, index=False)
        pos = int(frame["isFraud"].sum()) if "isFraud" in frame else 0
        logger.info("Wrote %s (%d rows, %d fraud)", out, len(frame), pos)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run()
