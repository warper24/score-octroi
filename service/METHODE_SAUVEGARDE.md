# Méthode de sauvegarde et réutilisation du modèle

Objectif: sérialiser, versionner et réutiliser le modèle pour un déploiement reproductible.

1) Librairies utilisées
- xgboost (modèle, format JSON)
- scikit-learn (prétraitements, OneHotEncoder)
- joblib (sérialisation du préprocesseur)
- pandas, numpy (manipulation des données)
- pyarrow (Parquet)
- pip freeze (inventaire exact des versions)

Un inventaire complet est stocké dans artifacts/model_vX.Y.Z/requirements.freeze.txt.

2) Artefacts générés (service/artifacts/model_vX.Y.Z/)
- model_xgb.json: modèle XGBoost sérialisé
- preprocessor.pkl: SimplePreprocessor (indication, exclusion, OHE)
- model_meta.json: métadonnées (type, version, threshold, variables)
- schema.json: dtypes d’entrée attendus
- reference.parquet: échantillon de référence out-of-sample (+ score)
- metrics.json: AUC, Gini, précision, rappel, PSI
- requirements.freeze.txt: versions exactes des libs
- VERSION: numéro de version
- SHA256SUMS: empreintes d’intégrité des fichiers

3) Procédure de sauvegarde (commandes)
- Entraîner et sauvegarder:
  python -m score_oc.cli --data-path .\data\tp_score_base0_anonymized.csv --date-col Gen_Demande --start-model 2014-05-01 --end-model 2015-12-01 --start-monitoring 2016-01-01 --end-monitoring 2016-08-01 --version 1.0.0
- Versionner dans Git:
  git add service/artifacts/model_v1.0.0 && git commit -m "Artifacts v1.0.0"
  git tag -a v1.0.0 -m "Model v1.0.0" && git push --tags

4) Modalités de réutilisation (inférence batch)
Exemple minimal pour charger et scorer un DataFrame brut:
```python
from pathlib import Path
import joblib, pandas as pd
from xgboost import XGBClassifier
from score_oc.config import VAR_MODEL, TARGET
from score_oc.preprocessing import SimplePreprocessor

art_dir = Path("service/artifacts/model_v1.0.0")
model = XGBClassifier()
model.load_model(art_dir / "model_xgb.json")
preproc: SimplePreprocessor = joblib.load(art_dir / "preprocessor.pkl")

df_raw = pd.read_csv("data/tp_score_base0_anonymized.csv")
X = preproc.transform(df_raw)
scores = model.predict_proba(X[VAR_MODEL])[:, 1]
```

5) Modalités de réutilisation (API)
- Lancer l’API (existant: service/api.py) puis envoyer un JSON de lignes brutes, l’API applique preprocessor + modèle.

6) Virtualisation / Containerisation (optionnel)
Dockerfile minimal (Python slim), copie des artefacts et de service/api.py, exécution via uvicorn. Les dépendances proviennent de requirements.freeze.txt pour garantir la reproductibilité.

7) Contrôles de conformité
- Intégrité: vérifier SHA256SUMS
- Compatibilité: vérifier que X.columns ⊇ VAR_MODEL avant prédiction
- Drift: comparer les scores/variables prod au reference.parquet (PSI)

8) Smoke test
Charger 10 lignes représentatives et valider:
- absence d’erreur de transform(),
- présence de toutes les VAR_MODEL,
- distribution des scores plausible (0..1).