from pathlib import Path
import json
import pandas as pd
from xgboost import XGBClassifier

ART_DIR = Path("service/artifacts/model_v1.0.0")

def test_artifacts_exist():
    for f in ["model_xgb.json", "preprocessor.pkl", "metrics.json", "schema.json", "reference.parquet"]:
        assert (ART_DIR / f).exists(), f"missing {f}"

def test_model_can_score_reference():
    model = XGBClassifier()
    model.load_model(ART_DIR / "model_xgb.json")
    ref = pd.read_parquet(ART_DIR / "reference.parquet")
    X = ref.drop(columns=[c for c in ["score"] if c in ref.columns])
    proba = model.predict_proba(X)[:,1]
    assert len(proba) == len(X)
    assert float(proba.min()) >= 0.0 and float(proba.max()) <= 1.0

def test_metrics_thresholds():
    m = json.loads((ART_DIR / "metrics.json").read_text())
    assert m["auc"] >= 0.70
    assert m["psi"] <= 0.2