## Méthode de sauvegarde du modèle (C5.3.1)

1. Entraînement dans notebooks/notebooks/modelling.ipynb.
2. Packaging: exécuter le script scripts/package_model.py (incrémente VERSION si changement).
3. Artifacts générés: service/artifacts/model_vX.Y.Z/
   - model_xgb.json : modèle XGBoost sérialisé (format natif JSON).
   - model_meta.json : variables, seuil, métriques, version.
   - schema.json : dtypes attendus.
   - reference.parquet : échantillon référence pour monitoring.
   - requirements.freeze.txt : versions exactes des libs.
   - VERSION : version sémantique.
   - SHA256SUMS : intégrité.
4. Réutilisation: service/inference.py charge le dernier dossier model_v*. API FastAPI (service/api.py).
5. Versioning: Git tag, ex: git tag v1.0.0 && git push --tags.
6. Containerisation: docker build -t bestcredit-score:1.0.0 . puis docker run -p 8000:8000 bestcredit-score:1.0.0.
7. Appel prédiction:
   POST http://localhost:8000/predict
   {
     "data": {
       "Age_Tit": 45,
       "Ressource": 2500.0,
       "...": 123.0
     }
   }

La méthode identifie les librairies (requirements + freeze), la modalité de sauvegarde (répertoire versionné + hash), et la réutilisation (module inference + API + container).