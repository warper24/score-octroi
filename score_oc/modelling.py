from typing import Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
from .metrics import gini_from_auc, compute_psi

def make_splits(df_ohe: pd.DataFrame, var_model: list[str], target: str,
                test_size: float = 0.30, random_state: int = 42):
    X = df_ohe[var_model].copy()
    y = df_ohe[target].copy()
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def train_xgb_simple(params: Dict, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba_test)
    gini = gini_from_auc(auc)
    prec = precision_score(y_test, y_pred_test)
    rec = recall_score(y_test, y_pred_test)
    psi = compute_psi(y_proba_train, y_proba_test, n_bins=10)
    return {"auc": float(auc), "gini": float(gini), "precision": float(prec), "recall": float(rec), "psi": float(psi)}
