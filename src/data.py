"""Dataset loading and the deterministic train/hold-out split shared by every stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ID_COLUMN, RANDOM_STATE, RAW_FEATURES, TARGET, TEST_SIZE


@dataclass(frozen=True)
class DatasetSplits:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_test: pd.Series


def load_dataset(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python data/generate_dataset.py` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in [ID_COLUMN, *RAW_FEATURES, TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df = df.dropna(subset=[TARGET]).drop_duplicates(subset=ID_COLUMN, keep="first")
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE
) -> DatasetSplits:
    """Stratified split. The hold-out set is used exactly once, after model selection."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    return DatasetSplits(
        X_train=train_df[RAW_FEATURES].reset_index(drop=True),
        X_test=test_df[RAW_FEATURES].reset_index(drop=True),
        y_train=train_df[TARGET].reset_index(drop=True),
        y_test=test_df[TARGET].reset_index(drop=True),
        ids_train=train_df[ID_COLUMN].reset_index(drop=True),
        ids_test=test_df[ID_COLUMN].reset_index(drop=True),
    )


def load_splits(path: Path | str, seed: int = RANDOM_STATE) -> DatasetSplits:
    return split_dataset(load_dataset(path), seed=seed)
