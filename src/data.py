from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLS = ["years_experience", "salary"]


def load_salary_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def to_xy(
    df: pd.DataFrame,
    feature_cols: List[str] = ["years_experience"],
    target_col: str = "salary",
) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y
