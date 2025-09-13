from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Any, Dict
import os
import joblib
import pandas as pd
from xgboost import XGBClassifier
from score_oc.config import VAR_MODEL
from score_oc.preprocessing import SimplePreprocessor
from pathlib import Path

ART_VERSION = os.getenv("ARTIFACT_VERSION", "1.0.0")
ART_DIR = Path(__file__).parent / "artifacts" / f"model_v{ART_VERSION}"

app = FastAPI(title="Score Octroi API", version=ART_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Static (front)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# Load artifacts at startup
model = XGBClassifier()
model.load_model(ART_DIR / "model_xgb.json")
preprocessor: SimplePreprocessor = joblib.load(ART_DIR / "preprocessor.pkl")

class Records(BaseModel):
    rows: List[Dict[str, Any]]

@app.get("/health")
def health():
    return {"status": "ok", "version": ART_VERSION}

@app.post("/predict")
def predict(payload: Records):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows is empty")
    df_raw = pd.DataFrame(payload.rows)
    X = preprocessor.transform(df_raw)
    missing = [c for c in VAR_MODEL if c not in X.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features after preprocessing: {missing}")
    proba = model.predict_proba(X[VAR_MODEL])[:, 1]
    return {"scores": proba.tolist(), "n": len(proba)}