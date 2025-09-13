import argparse
import json
from pathlib import Path
import pandas as pd

# Supporte: python -m score_oc.cli (imports relatifs) ET python .\score_oc\cli.py (fallback absolu)
try:
    from .config import VAR_MODEL, TARGET, VERSION, ARTIFACTS_DIR, XGB_PARAMS
    from .data import load_raw, select_columns, ensure_target
    from .selection import filter_period
    from .features import feature_engineering
    from .preprocessing import SimplePreprocessor
    from .modelling import make_splits, train_xgb_simple, evaluate_model
    from .artifacts import save_xgb_artifacts
except ImportError:
    import sys
    from pathlib import Path as _Path
    sys.path.append(str(_Path(__file__).resolve().parents[1]))
    from score_oc.config import VAR_MODEL, TARGET, VERSION, ARTIFACTS_DIR, XGB_PARAMS
    from score_oc.data import load_raw, select_columns, ensure_target
    from score_oc.selection import filter_period
    from score_oc.features import feature_engineering
    from score_oc.preprocessing import SimplePreprocessor
    from score_oc.modelling import make_splits, train_xgb_simple, evaluate_model
    from score_oc.artifacts import save_xgb_artifacts

def run_pipeline(data_path: Path, date_col: str | None,
                 start: str | None, end: str | None,
                 start_model: str | None, end_model: str | None,
                 start_monitoring: str | None, end_monitoring: str | None,
                 version: str | None) -> None:
    # 1) Data
    df_raw = load_raw(str(data_path))
    df_raw = feature_engineering(df_raw)

    # 1bis) Fenêtres temporelles
    res = filter_period(
        df_raw, date_col=date_col, start=start, end=end,
        start_model=start_model, end_model=end_model,
        start_monitoring=start_monitoring, end_monitoring=end_monitoring
    )
    if isinstance(res, tuple):
        df_model, df_ref = res
    else:
        df_model, df_ref = res, None

    df_model = select_columns(df_model)
    ensure_target(df_model, TARGET)
    if df_ref is not None and not df_ref.empty:
        df_ref = select_columns(df_ref)

    # 2) Preprocessing: fit sur la fenêtre “modélisation”, transform sur les deux
    preproc = SimplePreprocessor(rare_threshold=0.05).fit(df_model, target=TARGET)
    df_ohe = preproc.transform(df_model)
    df_ref_ohe = preproc.transform(df_ref) if df_ref is not None and not df_ref.empty else None

    # 3) Vérification colonnes modèle
    missing = [c for c in VAR_MODEL if c not in df_ohe.columns]
    if missing:
        raise ValueError(f"Colonnes du modèle absentes après preprocessing: {missing}")

    # 4) Splits sur la fenêtre “modélisation”
    X_train, X_test, y_train, y_test = make_splits(df_ohe, VAR_MODEL, TARGET)

    # 5) Modelling (XGB simple)
    model = train_xgb_simple(XGB_PARAMS, X_train, y_train)

    # 6) Eval (seuil = 0.5)
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    metrics_out = dict(metrics, threshold=0.5)

    # 7) Save artifacts (versioning) + jeu de référence (monitoring) si fourni
    out_dir = save_xgb_artifacts(
        model=model,
        preprocessor=preproc,
        X_train=X_train,
        X_test=X_test,
        var_model=VAR_MODEL,
        version=version or VERSION,
        base_dir=ARTIFACTS_DIR,
        reference_df=df_ref_ohe
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    print(f"Artifacts: {out_dir.resolve()}")
    print(f"Metrics: {json.dumps(metrics_out, indent=2)}")

def main():
    ap = argparse.ArgumentParser(description="Score Octroi - pipeline d'entraînement XGBoost")
    ap.add_argument("--data-path", type=Path, required=True, help="Chemin du CSV source (ex: .\\data\\model_data.csv)")
    ap.add_argument("--date-col", type=str, default=None, help="Nom de la colonne date (optionnel)")
    # Mode legacy (unique fenêtre)
    ap.add_argument("--start", type=str, default=None, help="Date de début unique (YYYY-MM-DD, optionnel)")
    ap.add_argument("--end", type=str, default=None, help="Date de fin unique (YYYY-MM-DD, optionnel)")
    # Double fenêtre (recommandé)
    ap.add_argument("--start-model", type=str, default=None, help="Début fenêtre modélisation (YYYY-MM-DD)")
    ap.add_argument("--end-model", type=str, default=None, help="Fin fenêtre modélisation (YYYY-MM-DD)")
    ap.add_argument("--start-monitoring", type=str, default=None, help="Début fenêtre monitoring (YYYY-MM-DD)")
    ap.add_argument("--end-monitoring", type=str, default=None, help="Fin fenêtre monitoring (YYYY-MM-DD)")
    ap.add_argument("--version", type=str, default=None, help="Override de la version des artefacts")
    args = ap.parse_args()
    run_pipeline(args.data_path, args.date_col, args.start, args.end,
                 args.start_model, args.end_model, args.start_monitoring, args.end_monitoring,
                 args.version)

if __name__ == "__main__":
    main()