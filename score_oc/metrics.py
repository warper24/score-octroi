import numpy as np
import pandas as pd

def gini_from_auc(auc: float) -> float:
    return 2 * float(auc) - 1

def compute_psi(expected, actual, n_bins=10, eps=1e-6) -> float:
    exp = pd.Series(expected).clip(0, 1)
    act = pd.Series(actual).clip(0, 1)
    q = min(n_bins, max(2, exp.nunique()))
    _, bins = pd.qcut(exp, q=q, retbins=True, duplicates='drop')
    bins[0] = min(bins[0], exp.min(), act.min())
    bins[-1] = max(bins[-1], exp.max(), act.max())
    exp_bins = pd.cut(exp, bins=bins, include_lowest=True)
    act_bins = pd.cut(act, bins=bins, include_lowest=True)
    exp_rate = (exp_bins.value_counts(sort=False) / len(exp)).replace(0, eps)
    act_rate = (act_bins.value_counts(sort=False) / len(act)).replace(0, eps)
    psi = ((exp_rate - act_rate) * np.log(exp_rate / act_rate)).sum()
    return float(psi)