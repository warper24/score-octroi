import json
from pathlib import Path

def test_metrics_thresholds():
    mpath = Path("service/artifacts/model_v1.0.0/metrics.json")
    data = json.loads(mpath.read_text(encoding="utf-8"))
    assert data["auc"] >= 0.65
    assert data["gini"] >= 0.30
    assert 0.0 <= data["psi"] < 0.2