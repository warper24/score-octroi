from pathlib import Path
import joblib
from xgboost import XGBClassifier

def test_artifacts_load():
    art = Path("service/artifacts/model_v1.0.0")
    assert (art / "model_xgb.json").exists()
    assert (art / "preprocessor.pkl").exists()
    model = XGBClassifier()
    model.load_model(art / "model_xgb.json")
    preproc = joblib.load(art / "preprocessor.pkl")
    assert hasattr(preproc, "transform")