import pandas as pd

# Fenêtre de référence par défaut (fixe)
REF_START_DEFAULT = "2016-01-01"
REF_END_DEFAULT = "2016-08-01"

def filter_period(
    df: pd.DataFrame,
    date_col: str | None = None,
    start: str | None = None,
    end: str | None = None,
    fmt: str | None = None,
    start_model: str | None = None,
    end_model: str | None = None,
    start_monitoring: str | None = None,
    end_monitoring: str | None = None,
):
    """
    Deux modes:
    - Mode simple (legacy): start/end -> retourne un seul DF filtré.
    - Mode double fenêtre: start_model/end_model et une fenêtre de référence (par défaut 2016-01-01 à 2016-08-01)
      -> retourne (df_model, df_ref). Les bornes de référence peuvent être surchargées via
      start_monitoring/end_monitoring si souhaité.
    """
    if not date_col or date_col not in df.columns:
        # Pas de colonne date ou pas de filtre -> comportement inchangé
        if any([start_model, end_model, start_monitoring, end_monitoring]):
            return df.copy(), df.iloc[0:0].copy()
        if start or end:
            return df.copy()
        return df.copy()

    d = pd.to_datetime(df[date_col], format=fmt, errors="coerce")

    def build_mask(s: str | None, e: str | None):
        m = pd.Series(True, index=df.index)
        if s:
            m &= d >= pd.to_datetime(s, format=fmt, errors="coerce")
        if e:
            m &= d <= pd.to_datetime(e, format=fmt, errors="coerce")
        return m

    # Mode double-fenêtre si au moins une borne “model” ou “monitoring” est fournie
    if any([start_model, end_model, start_monitoring, end_monitoring]):
        mask_model = build_mask(start_model, end_model)
        # Fenêtre de référence fixe par défaut si non fournie
        ref_start = start_monitoring or REF_START_DEFAULT
        ref_end = end_monitoring or REF_END_DEFAULT
        mask_ref = build_mask(ref_start, ref_end)
        return df.loc[mask_model].copy(), df.loc[mask_ref].copy()

    # Mode simple
    mask = build_mask(start, end)
    return df.loc[mask].copy()