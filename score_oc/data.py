import pandas as pd
from .config import DATA_COLUMNS, TARGET

def load_raw(csv_path: str, low_memory: bool = False) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=low_memory)
    return df

def select_columns(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or DATA_COLUMNS
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans la source: {missing}")
    return df[cols].copy()

def ensure_target(df: pd.DataFrame, target: str = TARGET) -> None:
    if target not in df.columns:
        raise ValueError(f"Cible '{target}' absente du DataFrame.")
    if df[target].nunique() < 2:
        raise ValueError(f"Cible '{target}' doit contenir 2 classes.")