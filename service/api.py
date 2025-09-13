from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from xgboost import Booster   # CHANGED
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

# Static (front)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

# Load artifacts at startup
booster = Booster()                         # CHANGED
booster.load_model(str(ART_DIR / "model_xgb.json"))  # CHANGED
preprocessor: SimplePreprocessor = joblib.load(ART_DIR / "preprocessor.pkl")

@app.get("/health")
def health():
    return {"status": "ok", "version": ART_VERSION}

@app.post("/predict")
def predict(records: list[dict]):
    if not isinstance(records, list) or len(records) == 0:
        raise HTTPException(status_code=400, detail="Body must be a non-empty JSON array")
    df = pd.DataFrame.from_records(records)
    X = preprocessor.transform(df)
    missing = [c for c in VAR_MODEL if c not in X.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features after preprocessing: {missing}")
    # CHANGED: use Booster inplace_predict (probas pour binary:logistic)
    prob = booster.inplace_predict(X[VAR_MODEL])
    return {"scores": list(map(float, prob)), "count": int(len(prob))}