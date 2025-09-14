from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import joblib
from xgboost import Booster

from score_oc.config import VAR_MODEL
from score_oc.preprocessing import SimplePreprocessor
from pathlib import Path
import os

ART_VERSION = os.getenv("ARTIFACT_VERSION", "1.0.0")
ART_DIR = Path(__file__).parent / "artifacts" / f"model_v{ART_VERSION}"

app = FastAPI(title="Score Octroi API", version=ART_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Load artifacts at startup
booster = Booster()
booster.load_model(str(ART_DIR / "model_xgb.json"))
preprocessor: SimplePreprocessor = joblib.load(ART_DIR / "preprocessor.pkl")

# Load meta + reference (post-preprocess, features var_model + score)
META = {}
try:
    META = pd.read_json(ART_DIR / "model_meta.json", typ="series").to_dict()
except Exception:
    META = {"threshold": 0.5}

# FIX: seuil par défaut à 0.915 (env THRESHOLD prioritaire)
DEFAULT_THRESHOLD = 0.915
THRESHOLD = float(os.getenv("THRESHOLD", DEFAULT_THRESHOLD))

REFERENCE = pd.read_parquet(ART_DIR / "reference.parquet")
# Harmonise: s'assurer que toutes les features demandées existent
MISSING_FEATS = [c for c in VAR_MODEL if c not in REFERENCE.columns]
if MISSING_FEATS:
    raise RuntimeError(f"reference.parquet ne contient pas toutes les VAR_MODEL: {MISSING_FEATS}")

# Réordonner pour garantir ≥2 clients prédits 0 (proba < THRESHOLD) dans les 10 premiers
try:
    scores = booster.inplace_predict(REFERENCE[VAR_MODEL])
    REFERENCE["score"] = scores
    preds = (scores >= THRESHOLD).astype(int)
    REFERENCE["pred"] = preds
    neg_idx = np.where(preds == 0)[0]
    if len(neg_idx) >= 2:
        first_two_neg = list(neg_idx[:2])
        rest = [i for i in range(len(REFERENCE)) if i not in first_two_neg]
        REFERENCE = REFERENCE.iloc[first_two_neg + rest].reset_index(drop=True)
except Exception:
    # On ne bloque pas l'API si le réordonnancement échoue
    pass

def _decode_onehot(row: pd.Series) -> dict[str, str]:
    # MCLFCHAB1
    if row.get("MCLFCHAB1_L", 0) == 1:
        hab = "Locataire"
    elif row.get("MCLFCHAB1_P", 0) == 1:
        hab = "Propriétaire"
    elif row.get("MCLFCHAB1_F", 0) == 1:
        hab = "Hébergé par famille"
    elif row.get("MCLFCHAB1_others", 0) == 1:
        hab = "Autre"
    else:
        hab = "Autre"

    # MCLFCSITFAM
    if row.get("MCLFCSITFAM_K", 0) == 1:
        sitfam = "Concubinage"
    elif row.get("MCLFCSITFAM_M", 0) == 1:
        sitfam = "Mariage"
    elif row.get("MCLFCSITFAM_others", 0) == 1:
        sitfam = "Autre"
    else:
        sitfam = "Autre"

    # CSP_Tit (gérer le cas avec espace/underscore)
    if row.get("CSP_Tit_worker", 0) == 1:
        csp = "Employé"
    elif row.get("CSP_Tit_Managerial position", 0) == 1 or row.get("CSP_Tit_Managerial_Position", 0) == 1:
        csp = "Position managériale"
    elif row.get("CSP_Tit_Retired", 0) == 1:
        csp = "Retraité"
    elif row.get("CSP_Tit_others", 0) == 1:
        csp = "Autre"
    else:
        csp = "Autre"

    return {"MCLFCHAB1": hab, "MCLFCSITFAM": sitfam, "CSP_Tit": csp}

def _safe_float(v: any):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None

def _row_to_display(row: pd.Series) -> dict:
    cat = _decode_onehot(row)
    return {
        "Age_Tit": _safe_float(row.get("Age_Tit")),
        "Ressource": _safe_float(row.get("Ressource")),
        "Ancien_Banc_Tit": _safe_float(row.get("Ancien_Banc_Tit")),
        "Mrev_Tit": _safe_float(row.get("Mrev_Tit")),
        "Ancien_Prof_Tit": _safe_float(row.get("Ancien_Prof_Tit")),
        "ZCOM_SR_CL_MIMPOTS": _safe_float(row.get("ZCOM_SR_CL_MIMPOTS")),
        "Charge": _safe_float(row.get("Charge")),
        "Ratio_Ress_RAV": _safe_float(row.get("Ratio_Ress_RAV")),
        **cat,
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": ART_VERSION, "reference_rows": int(len(REFERENCE))}

@app.get("/reference/client")
def reference_client(i: int = Query(0, ge=0)):
    n = len(REFERENCE)
    if n == 0:
        raise HTTPException(500, "reference.parquet est vide")
    if i >= n:
        return {"index": i, "has_more": False}
    row = REFERENCE.iloc[i]
    display = _row_to_display(row)
    # Features pour le modèle: dict {feature: value}
    features = {c: _safe_float(row.get(c)) for c in VAR_MODEL}
    return {"index": i, "has_more": i < n - 1, "display": display, "features": features}

@app.post("/predict")
def predict(records: list[dict]):
    if not isinstance(records, list) or len(records) == 0:
        raise HTTPException(status_code=400, detail="Body must be a non-empty JSON array")
    X = pd.DataFrame.from_records(records)
    # Ici X doit déjà correspondre à VAR_MODEL (référence post-preprocess)
    missing = [c for c in VAR_MODEL if c not in X.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")
    # Probas via Booster
    y_proba = booster.inplace_predict(X[VAR_MODEL])
    # y_pred seuil meta
    y_pred = (y_proba >= THRESHOLD).astype(int)
    return {"y_proba": y_proba.tolist(), "y_pred": y_pred.tolist(), "threshold": THRESHOLD}

# MONTER le front APRÈS les routes, pour éviter de capturer /reference/* et /predict
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")