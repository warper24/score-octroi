import pandas as pd
import numpy as np

def annee_complete_prof_val(v: float):
    if pd.isna(v):
        return np.nan
    return 1900 + v if v > 23 else 2000 + v

def annee_complete_banc_val(v: float):
    if pd.isna(v):
        return np.nan
    return 1900 + v if v > 23 else 2000 + v

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates demande / activité
    df["Gen_Demande"] = df["TDPRDCREAT"].astype(str).str[0:6]
    df["Gen_Active"] = np.where(df["ACTIVITE_DATE_min"].isin([99999999]), 0, df["ACTIVITE_DATE_min"].astype(str).str[:-2])
    df["Gen_Demande"] = pd.to_datetime(df["Gen_Demande"], errors = 'coerce', format = '%Y%m')
    df["Gen_Active"] = pd.to_datetime(df["Gen_Active"], errors = 'coerce', format = '%Y%m')
    df['Horizon_Activ'] = (df["Gen_Active"].dt.year - df["Gen_Demande"].dt.year) * 12 + (df["Gen_Active"].dt.month - df["Gen_Demande"].dt.month)
    # Revenus titulaires / conjoint
    df["Mrev_Tit"] = np.where(df["MCLICCSP__1"].isin([89, 92, 94, 95, 98, 0]), 0, df["MCLIMRESS__1"])
    mrev_cj = np.where(df["MCLICCSP__2"].isin([89, 92, 94, 95, 98, 0]), 0, df["MCLIMRESS__2"])
    df["Mrev_Cj"] = mrev_cj

    # Age
    df["Dnais_Tit"] = pd.to_datetime(df["MCLIDNAIS__1"].astype(str).str[6:], format="%Y", errors="coerce")
    df["Age_Tit"] = df["Gen_Demande"].dt.year - df["Dnais_Tit"].dt.year

    # Ancienneté professionnelle / bancaire (coercition numérique + vectorisé)
    v_prof = pd.to_numeric(df["MCLICANCPROF__1"].replace("**", np.nan), errors="coerce")
    anc_prof_year = np.where(pd.notna(v_prof), np.where(v_prof > 23, 1900 + v_prof, 2000 + v_prof), np.nan)
    df["Ancprof_Tit"] = anc_prof_year
    df["Ancien_Prof_Tit"] = pd.to_numeric(df["Gen_Demande"].dt.year - df["Ancprof_Tit"], errors="coerce")

    v_banc = pd.to_numeric(df["MBQRBOUVCPTE__1"], errors="coerce")
    annee_banc = np.where(pd.notna(v_banc), np.where(v_banc > 23, 1900 + v_banc, 2000 + v_banc), np.nan)
    df["Ancbanc_Tit"] = annee_banc
    df["Ancien_Banc_Tit"] = pd.to_numeric(df["Gen_Demande"].dt.year - df["Ancbanc_Tit"], errors="coerce")

    # Ressource / Charge / RAV
    df["Ressource"] = pd.to_numeric(
        df["MCLFMALLOCFAM"].fillna(0)
        + df["MCLFMALLOCLOG"].fillna(0)
        + df["Mrev_Tit"].fillna(0)
        + pd.Series(mrev_cj, index=df.index).fillna(0),
        errors="coerce",
    )
    df["Charge"] = pd.to_numeric(
        df["MCLFMFRAISGARD"].fillna(0)
        + df["MCPFMMENSRES1"].fillna(0)
        + df["MCPFMMENSRES2"].fillna(0),
        errors="coerce",
    )
    df["RAV"] = df["Ressource"] - df["Charge"]
    df["Ratio_Ress_RAV"] = df["Ressource"] / df["RAV"]

    # Défaut B et cible BP
    df["Defaut_B"] = 0
    for var in ["HMIN_R3_min", "HMIN_CTX_min"]:
        df["TEMP"] = df[var].apply(lambda x: 1 if 0 < x < 13 else 0)
        df["Defaut_B"] = df[["Defaut_B", "TEMP"]].max(axis=1)
    df.drop(columns=["TEMP"], inplace=True)
    df["BP"] = np.where(df["Defaut_B"] == 1, 0, 1)

    # Décision finale et flags
    conditions_Decision_Finale = [
        df["TDPRCPOSA"].isin(["SAN", "ENC", "FIN", "RET", "SOL", "INS"]),
        df["TDPRCPOSA"].isin(["ANN", "SS", "PRA"]),
        df["TDPRCPOSA"].isin(["REF", "RAG"]),
    ]
    values = ["ACP", "SS", "REF"]
    df["Decision_Finale"] = np.select(conditions_Decision_Finale, values, None)
    df["ACP"] = (df["Decision_Finale"] == "ACP").astype(int)
    df["SS"] = (df["Decision_Finale"] == "SS").astype(int)
    df["REF"] = (df["Decision_Finale"] == "REF").astype(int)
    df["Flag_Finance"] = (df["Decision_Finale"] == "ACP").astype(int)
    df["Fraudeur"] = df["MOUCH_ATENA_FRAUDCODE"].isin(["CG1", "CG2"]).astype(int)
    df["Flag_Actif"] = (~df["ACTIVITE_DATE_min"].isin([99999999])).astype(int)

    # CSP_Tit
    conditions_CSP_Tit = [
        df["MCLICCSP__1"].isin([20, 29]),
        df["MCLICCSP__1"].isin([1]),
        df["MCLICCSP__1"].isin([10, 11]),
        df["MCLICCSP__1"].isin([15]),
        df["MCLICCSP__1"].isin([19]),
        df["MCLICCSP__1"].isin([25, 64, 72, 86, 80, 81, 68, 73, 63, 66, 71]),
        df["MCLICCSP__1"].isin([30, 31, 55, 40, 41, 48, 28]),
        df["MCLICCSP__1"].isin([51, 50, 60, 70]),
        df["MCLICCSP__1"].isin([74]),
        df["MCLICCSP__1"].isin([90, 91]),
        df["MCLICCSP__1"].isin([92, 94, 98, 89]),
        df["MCLICCSP__1"].isin([96, 93]),
        df["MCLICCSP__1"].isin([85]),
        df["MCLICCSP__1"].isin([2]),
        df["MCLICCSP__1"].isin([97, 99]),
    ]
    values = [
        "liberal_profession",
        "Farmer",
        "Trader",
        "Artisan",
        "Stallholder",
        "worker",
        "Managerial position",
        "Employee",
        "Military",
        "Retired",
        "Unemployed",
        "Temporary contract",
        "Security",
        "Agricultural worker",
        "Executive",
    ]
    df["CSP_Tit"] = np.select(conditions_CSP_Tit, values, None)

    return df