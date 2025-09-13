import json, hashlib, pathlib
from typing import List, Dict, Any
import pandas as pd
import xgboost as xgb

ARTIFACT_ROOT = pathlib.Path(__file__).parent / "artifacts"

def latest_model_dir() -> pathlib.Path:
    dirs = sorted([p for p in ARTIFACT_ROOT.glob("model_v*") if p.is_dir()])
    if not dirs:
        raise FileNotFoundError("Aucun dossier de modèle versionné.")
    return dirs[-1]

def load_artifacts(model_dir: pathlib.Path = None):
    model_dir = model_dir or latest_model_dir()
    model_path = model_dir / "model_xgb.json"
    meta_path = model_dir / "model_meta.json"
    schema_path = model_dir / "schema.json"

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    return booster, meta, schema

def _validate_payload(payload: Dict[str, Any], schema: Dict[str, str]):
    missing = [k for k in schema if k not in payload]
    if missing:
        raise ValueError(f"Variables manquantes: {missing}")
    return {k: payload[k] for k in schema.keys()}

def predict_one(payload: Dict[str, Any]):
    booster, meta, schema = load_artifacts()
    row = _validate_payload(payload, schema["dtypes"])
    df = pd.DataFrame([row])
    # cast types
    for c, t in schema["dtypes"].items():
        if t == "int":
            df[c] = df[c].astype("int64")
        elif t == "float":
            df[c] = df[c].astype("float64")
        else:
            df[c] = df[c].astype(t)
    dmatrix = xgb.DMatrix(df[meta["var_model"]])
    proba = float(booster.predict(dmatrix)[0])
    pred = int(proba >= meta["threshold"])
    return {
        "probability_good_payer": proba,
        "prediction": pred,
        "threshold": meta["threshold"],
        "model_version": meta["model_version"]
    }

def predict_batch(rows: List[Dict[str, Any]]):
    booster, meta, schema = load_artifacts()
    processed = []
    for r in rows:
        processed.append(_validate_payload(r, schema["dtypes"]))
    df = pd.DataFrame(processed)
    for c, t in schema["dtypes"].items():
        if t == "int":
            df[c] = df[c].astype("int64")
        elif t == "float":
            df[c] = df[c].astype("float64")
    dmatrix = xgb.DMatrix(df[meta["var_model"]])
    probs = booster.predict(dmatrix)
    threshold = meta["threshold"]
    return [{
        "probability_good_payer": float(p),
        "prediction": int(p >= threshold)
    } for p in probs]