from pathlib import Path
import json, datetime, subprocess, hashlib
import pandas as pd
import joblib

def save_xgb_artifacts(model, preprocessor,
                       X_train: pd.DataFrame, X_test: pd.DataFrame,
                       var_model: list[str], version: str,
                       base_dir: Path, reference_df: pd.DataFrame | None = None) -> Path:
    out_dir = base_dir / f"model_v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Modèle + préprocesseur
    model.save_model(out_dir / "model_xgb.json")
    if preprocessor is not None:
        joblib.dump(preprocessor, out_dir / "preprocessor.pkl")

    # 2) Meta
    meta = {
        "model_type": "xgboost",
        "model_version": version,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "var_model": var_model,
        "threshold": 0.5
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # 3) Schéma
    dtype_map = {}
    for c in var_model:
        dt = str(X_train[c].dtype)
        dtype_map[c] = "int" if dt.startswith("int") else ("float" if dt.startswith("float") else "float")
    schema = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z", "dtypes": dtype_map, "primary_key": None}
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # 4) Jeu de référence (monitoring si fourni, sinon fallback test)
    df_ref = (reference_df[var_model].copy() if reference_df is not None and not reference_df.empty
              else X_test[var_model].copy())
    df_ref["score"] = model.predict_proba(df_ref[var_model])[:, 1]
    df_ref.to_parquet(out_dir / "reference.parquet")

    # 5) Freeze des dépendances
    req_text = subprocess.check_output(["pip", "freeze"], text=True)
    (out_dir / "requirements.freeze.txt").write_text(req_text, encoding="utf-8")

    # 6) Fichier VERSION
    (out_dir / "VERSION").write_text(version, encoding="utf-8")

    # 7) Hash d’intégrité
    hash_lines = []
    for p in out_dir.iterdir():
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            hash_lines.append(f"{h}  {p.name}")
    (out_dir / "SHA256SUMS").write_text("\n".join(hash_lines), encoding="utf-8")
    return out_dir