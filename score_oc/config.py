from pathlib import Path

VERSION = "1.0.0"
TARGET = "BP"
ARTIFACTS_DIR = Path("service") / "artifacts"

# Colonnes brutes utiles (depuis le notebook)
DATA_COLUMNS = [
    "ZCOM_SR_CL_MIMPOTS","Mrev_Tit","Ressource","Charge","Ancbanc_Tit",
    "Ancien_Banc_Tit","Ancien_Prof_Tit","Ratio_Ress_RAV","Age_Tit",
    "Flag_Actif","Fraudeur","Flag_Finance","Gen_Active","Horizon_Activ",
    "MCLFCHAB1","MCLFCSITFAM","CSP_Tit","BP","Gen_Demande"
]

# Variables du modèle (encoded, cohérent avec le notebook "Final Model XGBoost")
VAR_MODEL = [
    "Age_Tit","Ressource","Ancien_Banc_Tit","Mrev_Tit","Ancien_Prof_Tit",
    "ZCOM_SR_CL_MIMPOTS","Charge","Ratio_Ress_RAV",
    "MCLFCHAB1_L","CSP_Tit_worker","CSP_Tit_others","MCLFCHAB1_P","MCLFCHAB1_F",
    "MCLFCSITFAM_K","MCLFCSITFAM_M","CSP_Tit_Managerial position","MCLFCSITFAM_others",
    "CSP_Tit_Retired","MCLFCHAB1_others"
]

# Hyperparamètres XGB simples (comme dans le notebook)
XGB_PARAMS = dict(
    objective="binary:logistic",
    eval_metric="auc",
    n_estimators=100,
    learning_rate=0.2,
    max_depth=5,
    min_child_weight=0,
    subsample=1.0,
    colsample_bytree=0.8,
    gamma=5,
    random_state=42,
    n_jobs=-1,
)