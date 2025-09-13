import os, json, pathlib, hashlib, subprocess, datetime
import pandas as pd
import xgboost as xgb

# Inputs attendus: objets déjà créés dans le notebook et sauvegardés
# Recharger si besoin: le notebook peut exporter X_train, var_model, threshold, xgb_best

VERSION = "1.0.0"  # incrémenter (MAJOR.MINOR.PATCH)

ARTIF_ROOT = pathlib.Path("service/artifacts")
MODEL_DIR = ARTIF_ROOT / f"model_v{VERSION}"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hypothèse: notebook a déjà créé xgb_best, var_model, auc_xgb_test, gini_xgb_test
# Adapter si différent
from joblib import dump

def save_model(booster):
    model_path = MODEL_DIR / "model_xgb.json"
    booster.save_model(str(model_path))
    return model_path

def write_version():
    (MODEL_DIR / "VERSION").write_text(VERSION, encoding="utf-8")

def capture_requirements():
    req = subprocess.check_output(["pip","freeze"], text=True)
    (MODEL_DIR / "requirements.freeze.txt").write_text(req, encoding="utf-8")

def build_schema(feature_list):
    # Dans ce projet: toutes numériques
    dtypes = {f: "float" for f in feature_list}
    schema = {
        "generated_at": datetime.datetime.utcnow().isoformat()+"Z",
        "dtypes": dtypes,
        "primary_key": None
    }
    (MODEL_DIR / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

def save_meta(var_model, threshold, metrics: dict):
    meta = {
        "model_type": "xgboost",
        "model_version": VERSION,
        "created_at": datetime.datetime.utcnow().isoformat()+"Z",
        "var_model": var_model,
        "threshold": threshold,
        "metrics": metrics
    }
    (MODEL_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def copy_reference(reference_parquet_path: str):
    import shutil
    shutil.copy(reference_parquet_path, MODEL_DIR / "reference.parquet")

def hash_files():
    lines = []
    for p in MODEL_DIR.glob("*"):
        if p.is_file():
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.name}")
    (MODEL_DIR / "SHA256SUMS").write_text("\n".join(lines), encoding="utf-8")

def main():
    # Ces objets doivent être chargés (adapter si noms différents)
    from modelling_context import xgb_best, var_model, auc_xgb_test, gini_xgb_test  # exemple si vous créez un module
    threshold = 0.5
    save_model(xgb_best.get_booster())
    write_version()
    capture_requirements()
    build_schema(var_model)
    save_meta(var_model, threshold, {"auc_test": float(auc_xgb_test), "gini_test": float(gini_xgb_test)})
    copy_reference("service/artifacts/reference.parquet")
    hash_files()
    print(f"Packagé dans {MODEL_DIR.resolve()}")

if __name__ == "__main__":
    main()