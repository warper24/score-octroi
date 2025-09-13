import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from dataclasses import dataclass, field

def indication(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Age_Tit"] = np.where((df["Age_Tit"] < 18) | (df["Age_Tit"] > 85), df["Age_Tit"].median(), df["Age_Tit"])
    df["Ancien_Prof_Tit"] = np.where(df["Ancien_Prof_Tit"] > 55, 999999, df["Ancien_Prof_Tit"])
    df["ZCOM_SR_CL_MIMPOTS"] = np.where(df["ZCOM_SR_CL_MIMPOTS"] > 10000, df["ZCOM_SR_CL_MIMPOTS"].median(), df["ZCOM_SR_CL_MIMPOTS"])
    df["Mrev_Tit"] = np.where(df["Mrev_Tit"] > 100000, df["Mrev_Tit"].median(), df["Mrev_Tit"])
    df["Mrev_Tit"] = df["Mrev_Tit"].fillna(df["Mrev_Tit"].median())
    df["Ancien_Banc_Tit"] = df["Ancien_Banc_Tit"].fillna(df["Ancien_Banc_Tit"].median())
    df["Ancien_Prof_Tit"] = df["Ancien_Prof_Tit"].fillna(df["Ancien_Prof_Tit"].median())
    df["Ratio_Ress_RAV"] = df["Ratio_Ress_RAV"].fillna(df["Ratio_Ress_RAV"].median())
    df["Age_Tit"] = df["Age_Tit"].fillna(df["Age_Tit"].median())
    for col in ["MCLFCHAB1", "MCLFCSITFAM", "CSP_Tit"]:
        if col in df.columns:
            # Cast en string pour harmoniser avec l'encodage
            df[col] = df[col].astype("string").fillna("Missing")
    return df

def exclusion(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Flag_Finance" in df.columns:
        df = df[df["Flag_Finance"] == 1]
    if "Fraudeur" in df.columns:
        df = df[df["Fraudeur"] != 1]
    if "Flag_Actif" in df.columns:
        df = df[df["Flag_Actif"] != 0]
    if "Horizon_Activ" in df.columns:
        df = df[df["Horizon_Activ"] <= 4]
    return df

def iqr_clip_and_median(df: pd.DataFrame, target: str) -> pd.DataFrame:
    num_cols = df.select_dtypes(include='number').columns.tolist()
    num_cols = [c for c in num_cols if c != target]
    if not num_cols:
        return df.copy()
    num = df[num_cols]
    q1, q3 = num.quantile(0.25), num.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 2 * iqr, q3 + 2 * iqr
    out = df.copy()
    for c in num_cols:
        out[c] = out[c].clip(lower=lower[c], upper=upper[c])
    for c in num_cols:
        mask = (df[c] < lower[c]) | (df[c] > upper[c])
        if mask.any():
            out.loc[mask, c] = df[c].median()
    return out

def collapse_rare_categories(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    out = df.copy()
    cat_cols = out.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in cat_cols:
        s = out[col].astype("string").fillna("Missing")
        freq = s.value_counts(normalize=True, dropna=False)
        rare = freq.index[freq < threshold].tolist()
        if rare:
            s = s.where(~s.isin(rare), "others")
        out[col] = s.astype("string")
    return out

def one_hot_encode(df: pd.DataFrame):
    cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not cat_cols:
        return df.copy(), None, []
    df_cat = df[cat_cols].astype("string").fillna("Missing")
    enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    X_cat = enc.fit_transform(df_cat)
    encoded_cols = enc.get_feature_names_out(cat_cols)
    df_ohe = pd.DataFrame(X_cat, columns=encoded_cols, index=df.index)
    out = pd.concat([df.drop(columns=cat_cols), df_ohe], axis=1)
    return out, enc, cat_cols

def preprocess_df(df: pd.DataFrame, target: str):
    df = indication(df)
    tmp = iqr_clip_and_median(df, target=target)
    tmp = collapse_rare_categories(tmp, threshold=0.05)
    df_ohe, enc, _ = one_hot_encode(tmp)
    return df_ohe, enc

@dataclass
class SimplePreprocessor:
    rare_threshold: float = 0.05
    target_: str | None = None
    num_cols_: list = field(default_factory=list)
    cat_cols_: list = field(default_factory=list)
    lower_: dict = field(default_factory=dict)
    upper_: dict = field(default_factory=dict)
    medians_: dict = field(default_factory=dict)
    rare_sets_: dict = field(default_factory=dict)
    enc_: OneHotEncoder | None = None

    def fit(self, df: pd.DataFrame, target: str):
        self.target_ = target
        # 0) Indication puis exclusion
        df_ind = indication(df.copy())
        df_ind = exclusion(df_ind)

        # 1) Numériques
        num_cols = df_ind.select_dtypes(include="number").columns.tolist()
        self.num_cols_ = [c for c in num_cols if c != target]
        if self.num_cols_:
            num = df_ind[self.num_cols_]
            q1, q3 = num.quantile(0.25), num.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 2 * iqr, q3 + 2 * iqr
            self.lower_ = lower.to_dict()
            self.upper_ = upper.to_dict()
            self.medians_ = num.median().to_dict()

        # 2) Catégorielles (cast -> string, Missing, rares -> others)
        self.cat_cols_ = df_ind.select_dtypes(include=["object", "string"]).columns.tolist()
        self.rare_sets_.clear()
        if self.cat_cols_:
            df_cat = df_ind[self.cat_cols_].astype("string").fillna("Missing").copy()
            for col in self.cat_cols_:
                freq = df_cat[col].value_counts(normalize=True, dropna=False)
                rare = set(freq.index[freq < self.rare_threshold].tolist())
                self.rare_sets_[col] = rare
                if rare:
                    df_cat[col] = df_cat[col].where(~df_cat[col].isin(rare), "others")
            self.enc_ = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            self.enc_.fit(df_cat[self.cat_cols_])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = indication(df.copy())
        out = exclusion(out)

        # Numériques
        for c in self.num_cols_:
            if c not in out.columns:
                continue
            out[c] = out[c].clip(lower=self.lower_.get(c, out[c].min()), upper=self.upper_.get(c, out[c].max()))
            mask = (out[c] < self.lower_.get(c, -np.inf)) | (out[c] > self.upper_.get(c, np.inf))
            if mask.any():
                out.loc[mask, c] = self.medians_.get(c, out[c].median())

        # Catégorielles
        if self.cat_cols_:
            for col in self.cat_cols_:
                if col not in out.columns:
                    out[col] = pd.NA
            df_cat = out[self.cat_cols_].astype("string").fillna("Missing")
            for col in self.cat_cols_:
                rare = self.rare_sets_.get(col)
                if rare:
                    df_cat[col] = df_cat[col].where(~df_cat[col].isin(rare), "others")
            X_cat = self.enc_.transform(df_cat[self.cat_cols_])
            encoded_cols = self.enc_.get_feature_names_out(self.cat_cols_)
            df_ohe = pd.DataFrame(X_cat, columns=encoded_cols, index=out.index)
            out = pd.concat([out.drop(columns=self.cat_cols_, errors="ignore"), df_ohe], axis=1)

        return out