from typing import Tuple, Union
from portion.interval import Interval
from pandas.api.types import is_numeric_dtype, is_interval_dtype
from pandas.api.types import CategoricalDtype
import logging
import re
import numpy
import pandas
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import nbformat

logger = logging.getLogger(__name__)
__all__ = [
    "default",
    "default_and_ctx",
    "defaults_table",
    "_is_default_variable",
    "_default_sort_key",
    "select_and_sort_default_variables",
    "risk_tables",
    "compute_generation",
    "compute_horizon",
    "plot_gen_horizon",
    "risk_curve",
    "plot_horizon_activ",
    "plot_evolution_risk_rates",
    "get_horizon",
]
CHECK_MSG = "{} , please check the column : {}"
ERRORS_TYPE = ["raise", "ignore"]
ERRORS_MESSAGE = "errors parameter must be in ['raise', 'ignore']"

def check_target(array: pandas.Series, errors: str = "raise"):
    """Check target values
    Parameters
    ----------
    array : pandas.Series
    errors : str, optional (default "raise")
    """
    if errors not in ERRORS_TYPE:
        raise ValueError(ERRORS_MESSAGE)
    target_values = [numpy.nan, 0, 1]
    match_values = array.isin(target_values)
    if not all(match_values) and not all((array >= 0.0) & (array <= 1.0)):
        unexpected_values = array[~match_values].to_list()
        msg = "Target variable contains values not authorized : {}, must be in {}.".format(unexpected_values, str(target_values))
        if errors == "raise":
            raise ValueError(msg)
        elif errors == "ignore":
            return False
    return True

def check_weights(array: pandas.Series, errors: str = "raise"):
    """Check weight values
    Parameters
    ----------
    array : pandas.Series
    errors : str, optional (default "raise")
    """
    if errors not in ERRORS_TYPE:
        raise ValueError(ERRORS_MESSAGE)
    check_is_numerical(array)
    if not all(array > 0):
        unexpected_values = array[array <= 0].to_list()
        msg = "Weight variable contains values not authorized : {}, must be non zero and positive.".format(unexpected_values)
        if errors == "raise":
            raise ValueError(msg)
        elif errors == "ignore":
            return False
    return True

def safe_concat(series_list, ignore_index=False):
    ref_index = series_list[0].index
    ref_hash = pandas.util.hash_pandas_object(ref_index).values
    for i, series in enumerate(series_list[1:], start=1):
        if not (pandas.util.hash_pandas_object(series.index).values == ref_hash).all():
            raise ValueError("Cannot concatenate Series. Check if indexes of all Series are the same.")
    return pandas.concat(series_list, axis=1, ignore_index=ignore_index)

def frequency(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None) -> pandas.DataFrame:
    """Compute frequency of bad and good payer for every modality in the variable.
    Parameters
    ----------
    variable : pandas.Series
    risk_variable : pandas.Series
    weight_variable: pandas.Series
    Returns
    -------
    pandas.DataFrame
    """
    table = risk_cardinality(variable, risk_variable, weight_variable)
    table["FREQ_BAD_PAYER"] = table["NB_BAD_PAYER"] / table["NB_BAD_PAYER"].sum()
    table["FREQ_GOOD_PAYER"] = table["NB_GOOD_PAYER"] / table["NB_GOOD_PAYER"].sum()
    return table[["FREQ_BAD_PAYER", "FREQ_GOOD_PAYER"]]

def check_is_interval(array: pandas.Series, name: str = None, errors: str = "raise"):
    """Check if array contains pandas.Interval type.
    Parameters
    ----------
    array : pandas.Series
    name : str
    errors : str
    """
    if errors not in ERRORS_TYPE:
        raise ValueError(ERRORS_MESSAGE)
    if not all(isinstance(val, Interval) for val in array.unique()):
        msg = "Input is not interval : " + str(array.dtype)
        if name:
            msg = CHECK_MSG.format(msg, name)
        if errors == "raise":
            raise ValueError(msg)
        elif errors == "ignore":
            return False
    return True

def apply_style(table, total=True):
    style_precision = {
        "NB_APPLICANTS": "{:.0f}",
        "NB_GOOD_PAYER": "{:.0f}",
        "NB_BAD_PAYER": "{:.0f}",
        "FREQ_BAD_PAYER": "{0:.2%}",
        "FREQ_GOOD_PAYER": "{0:.2%}",
        "FREQ_TOTAL_APPLICANTS": "{0:.2%}",
        "RISK_RATE": "{0:.2%}",
        "ODDS": "{:.3f}",
        "WOE": "{:.3f}",
        "IV": "{:.2f}",
        "FREQ_CUM_BAD_PAYER": "{0:.2%}",
        "FREQ_CUM_GOOD_PAYER": "{0:.2%}",
        "P_GINI": "{0:.2%}",
        "GINI": "{0:.2%}",
    }
    if total:
        table.index = table.index.map(str)
        table.loc["TOTAL"] = table.sum(axis=0)
        table.loc["TOTAL", "WOE"] = None
        table.loc["TOTAL", "ODDS"] = None
        table.loc["TOTAL", "RISK_RATE"] = table.loc["TOTAL", "NB_BAD_PAYER"] / table.loc["TOTAL", "NB_APPLICANTS"]
        if "GINI" in table.columns.tolist():
            table.loc["TOTAL", "FREQ_CUM_BAD_PAYER"] = 1
            table.loc["TOTAL", "FREQ_CUM_GOOD_PAYER"] = 1
            table.loc["TOTAL", "GINI"] = None
    l = list(zip([0, 0.15, 0.4, 0.5, 0.6, 0.9, 1], ["darkgreen", "green", "palegreen", "white", "lightcoral", "red", "darkred"]))
    cmap_risk_rate = LinearSegmentedColormap.from_list("rg", l)
    columns = table.index.tolist()
    if total:
        columns = columns[:-1]
        _subset = [(columns, ["FREQ_BAD_PAYER"]), (columns, ["FREQ_GOOD_PAYER"]), (columns, ["FREQ_TOTAL_APPLICANTS"]), (columns, ["RISK_RATE"]), (columns, ["IV"])]
    else:
        _subset = [["FREQ_BAD_PAYER"], ["FREQ_GOOD_PAYER"], ["FREQ_TOTAL_APPLICANTS"], ["RISK_RATE"], ["IV"]]
    styler = (
        table.style.bar(height=80, color="indianred", vmin=0, vmax=1, align="left", subset=_subset[0])
        .bar(height=80, color="lightgreen", vmin=0, vmax=1, align="left", subset=_subset[1])
        .bar(height=80, color="lightblue", vmin=0, vmax=1, align="left", subset=_subset[2])
        .background_gradient(vmin=table["RISK_RATE"].min(), vmax=table["RISK_RATE"].max(), cmap=cmap_risk_rate, subset=_subset[3])
        .background_gradient(cmap="Purples", vmin=0, subset=_subset[4])
    )
    table = styler.format(style_precision, na_rep="")
    return table
def check_is_not_null(array: pandas.Series, name: str = None, errors: str = "raise"):
    """
    Check if there is no missing values in a array.
    Parameters
    ----------
    array : pandas.Series
    name: str
        the name of the array
    errors: str
        raise an error when array contain missing value, by default "raise"
    """
    if errors not in ERRORS_TYPE:
        raise ValueError(ERRORS_MESSAGE)
    if not all(pandas.notna(array.values)):
        msg = "Input contains NaN"
        if name:
            msg = CHECK_MSG.format(msg, name)
        if errors == "raise":
            raise ValueError(msg)
        elif errors == "ignore":
            return False
    return True
def odds(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None) -> pandas.DataFrame:
    """
    Compute odds for the variable (proportion of good / proportion of bad) for every modality.
    Parameters
    ----------
    variable : pandas.Series
    risk_variable : pandas.Series
    weight_variable : pandas.Series
    Returns
    -------
    table[["ODDS"]] : pandas.DataFrame
        Log-Odds for every modality in the variable
    """
    table = risk_cardinality(variable, risk_variable, weight_variable)
    table["ODDS"] = (table["NB_GOOD_PAYER"] / table["NB_APPLICANTS"]) / (table["NB_BAD_PAYER"] / table["NB_APPLICANTS"])
    return table[["ODDS"]]
def risk_cardinality(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None) -> pandas.DataFrame:
    """
    Count the number of total applicants, number of good and bad payer.
    Parameters
    ----------
    variable : pandas.Series
    risk_variable : pandas.Series
    weight_variable: pandas.Series, optional
    Returns
    -------
    table : pandas.DataFrame
        Absolute frequency for every modality in the variable
    """
    check_target(risk_variable)
    if weight_variable is None:
        weight_variable = pandas.Series(data=1, index=list(variable.index), name="WEIGHT")
    else:
        check_weights(weight_variable)
    data = safe_concat([variable, risk_variable, weight_variable])
    table = (
        data.groupby(variable.name)
        .agg(
            **{
                "NB_APPLICANTS": pandas.NamedAgg(column=weight_variable.name, aggfunc=lambda x: (data.loc[x.index, weight_variable.name]).sum()),
                "NB_GOOD_PAYER": pandas.NamedAgg(column=risk_variable.name, aggfunc=lambda x: (x * data.loc[x.index, weight_variable.name]).sum()),
                "NB_BAD_PAYER": pandas.NamedAgg(column=risk_variable.name, aggfunc=lambda x: ((1 - x) * data.loc[x.index, weight_variable.name]).sum()),
            }
        )
        .astype(float)
    )
    return table
def adjust_weight_of_evidence(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None) -> pandas.DataFrame:
    """Compute adjust weight of evidence.
    Parameters
    ----------
    variable : pandas.Series
        Variable to analyze
    risk_variable : pandas.Series
        Target that identifies good payer
    weight_variable : pandas.Series
        Weight variable
    Returns
    -------
    WOE : pandas.DataFrame
        WOE for every modality in the variable
    """
    table = risk_cardinality(variable, risk_variable, weight_variable)
    table["WOE"] = numpy.log(((table["NB_GOOD_PAYER"] + 0.5) / (table["NB_GOOD_PAYER"].sum() + 0.5)) / ((table["NB_BAD_PAYER"] + 0.5) / (table["NB_BAD_PAYER"].sum() + 0.5)))
    return table[["WOE"]]
def information_value(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None) -> pandas.DataFrame:
    """
    Compute information value.
    Parameters
    ----------
    variable : pandas.Series
    risk_variable : pandas.Series
    weight_variable : pandas.Series
    Returns
    -------
    table[["IV"]] : pandas.DataFrame
        IV for every modality in the variable
    """
    table = risk_cardinality(variable, risk_variable, weight_variable)
    woe = adjust_weight_of_evidence(variable, risk_variable, weight_variable)
    table["IV"] = ((table["NB_GOOD_PAYER"] / table["NB_GOOD_PAYER"].sum()) - (table["NB_BAD_PAYER"] / table["NB_BAD_PAYER"].sum())) * woe["WOE"]
    return table[["IV"]]
def cumulative_frequency(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None) -> pandas.DataFrame:
    """
    Compute cumulative frequency of good and bad payer
    Parameters
    ----------
    variable : pandas.Series
    risk_variable : pandas.Series
    Returns
    -------
    table : pandas.DataFrame
        Adds cumulative relative frequency column to absolute frequency table
    """
    table = risk_cardinality(variable, risk_variable, weight_variable)
    table["FREQ_CUM_BAD_PAYER"] = table["NB_BAD_PAYER"].cumsum() / table["NB_BAD_PAYER"].sum()
    table["FREQ_CUM_GOOD_PAYER"] = table["NB_GOOD_PAYER"].cumsum() / table["NB_GOOD_PAYER"].sum()
    return table[["FREQ_CUM_BAD_PAYER", "FREQ_CUM_GOOD_PAYER"]]
def partial_gini(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None) -> pandas.DataFrame:
    """
    Compute partial gini for every interval.
    Parameters
    ----------
    variable : pandas.Series
    risk_variable : pandas.Series
    Returns
    -------
    p_gini : pandas.DataFrame
        Partial gini table for each interval
    """
    freq = frequency(variable, risk_variable, weight_variable)
    cum_freq = cumulative_frequency(variable, risk_variable, weight_variable)
    p_gini = (cum_freq["FREQ_CUM_BAD_PAYER"] + cum_freq["FREQ_CUM_BAD_PAYER"].shift(1).fillna(0) - cum_freq["FREQ_CUM_GOOD_PAYER"] - cum_freq["FREQ_CUM_GOOD_PAYER"].shift(1).fillna(0)) * freq["FREQ_BAD_PAYER"]
    if p_gini.sum() < 0:
        p_gini = p_gini * -1
    return pandas.DataFrame(p_gini, columns=["P_GINI"])
def cramers_v(variable_1: pandas.Series, variable_2: pandas.Series) -> float:
    """
    Compute Cramer V correlation between two categorical variable.
    Parameters
    ----------
    variable_1 : pandas.Series
    variable_2 : pandas.Series
    Returns
    -------
    float
        Cramer V value
    """
    confusion_matrix = pandas.crosstab(variable_1.astype(str), variable_2.astype(str))
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return numpy.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
def marginal_analysis(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None, round_value: Union[int, None] = None, verbose: bool = False, style: bool = False) -> pandas.DataFrame:
    """Perform a marginal analysis : Counting, Frequency, Reject Rate, Risk Rate, Odds, WOE, IV and GINI
    Parameters
    ----------
    variable : pandas.Series
    risk_variable : pandas.Series
    weight_variable : pandas.Series, optional
    round_value : int, optional
    verbose : bool, optional
    style : bool, optional
    Returns
    -------
    table : pandas.DataFrame
        Marginal analysis table for each modality in variable
    """
    table = risk_cardinality(variable, risk_variable, weight_variable)
    table["FREQ_BAD_PAYER"] = table["NB_BAD_PAYER"] / table["NB_BAD_PAYER"].sum()
    table["FREQ_GOOD_PAYER"] = table["NB_GOOD_PAYER"] / table["NB_GOOD_PAYER"].sum()
    table["FREQ_TOTAL_APPLICANTS"] = table["NB_APPLICANTS"] / table["NB_APPLICANTS"].sum()
    table["RISK_RATE"] = table["NB_BAD_PAYER"] / (table["NB_GOOD_PAYER"] + table["NB_BAD_PAYER"])
    if not check_is_not_null(risk_variable, name=risk_variable.name, errors="ignore"):
        table["REJECT_RATE"] = 1 - ((table["NB_GOOD_PAYER"] + table["NB_BAD_PAYER"]) / (table["NB_APPLICANTS"]))
    table["ODDS"] = odds(variable, risk_variable, weight_variable)
    table["WOE"] = adjust_weight_of_evidence(variable, risk_variable, weight_variable)
    table["IV"] = information_value(variable, risk_variable, weight_variable)
    if ((isinstance(variable, pandas.CategoricalDtype) and is_interval_dtype(variable.dtypes.categories)) or (is_numeric_dtype(variable)) or check_is_interval(variable, errors="ignore")):
        table[["FREQ_CUM_BAD_PAYER", "FREQ_CUM_GOOD_PAYER"]] = cumulative_frequency(variable, risk_variable, weight_variable)
        table["P_GINI"] = partial_gini(variable, risk_variable, weight_variable)
        table["GINI"] = table["P_GINI"].cumsum()
    if verbose:
        if "P_GINI" in table.columns:
            print("TOTAL GINI = " + str(table["P_GINI"].sum()))
        print("TOTAL IV = " + str(table["IV"].sum()))
    if isinstance(round_value, (int, float)):
        return table.round(round_value)
    if style:
        styler = apply_style(table, total=True)
        return styler
    return table
def check_is_numerical(array: pandas.Series, name: str = None, errors: str = "raise"):
    """Check if the array is numerical (float or integer
    Parameters
    ----------
    array : pandas.Series
    name : str, optional
        by default None
    errors : str, optional
        by default "raise"
    """
    if errors not in ERRORS_TYPE:
        raise ValueError(ERRORS_MESSAGE)
    if not is_numeric_dtype(array):
        msg = "Input is not numerical : " + str(array.dtype)
        if name:
            msg = CHECK_MSG.format(msg, name)
        if errors == "raise":
            raise ValueError(msg)
        elif errors == "ignore":
            return False
    return True
def plot_risk_by_cut(variable: pandas.Series, risk_variable: pandas.Series, weight_variable: pandas.Series = None, bins=None, volume_max: int = None, risk_max: float = None, show_average: bool = True):
    """Plot risk rate and volumes for each modalitie,
    according to the selected binning method.
    Parameters
    ----------
    variable : pandas.Series
        Variable to analyze
    risk_variable : pandas.Series
        Risk target varaible
    weight_variable : pandas.Series
        Frequency weights
    bins : str, list, bool or int
        Bins to split the 'var' variable.
        if str: use of a cut avaliable in model
        if list: list of interval limits for the bins,
            for quantitative variables
        if False: plots each category separated,
            for qualitative variables
        if int: number of quantiles to split the data,
            for quantitative variables
    volume_max : int
        Upper limit of volume axis, in percentage from 0 to 100
        Used in ylim max input fro the left axis
    risk_max : float
        Upper limit of risk rate axis, in percentage from 0 to 100
        Used in ylim max input fro the right axis
    show_average : bool
        If True shows the avrega risk rate line
    Returns
    ----------
    fig, [ax1, ax2] : tuple
        figure and axis objects from matplotlib
    """
    if check_is_numerical(variable, errors="ignore"):
        if isinstance(bins, int):
            variable = pandas.qcut(variable, bins, duplicates="drop")
        elif isinstance(bins, list):
            bins = sorted(set([variable.min()] + bins + [variable.max()]))
            variable = pandas.cut(variable, bins, include_lowest=True)
    marginal_table = marginal_analysis(variable, risk_variable, weight_variable)
    index = [str(idx) for idx in marginal_table.index]
    risk_rate = [round(r, 3) for r in marginal_table.RISK_RATE]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=index, y=marginal_table.FREQ_TOTAL_APPLICANTS, secondary_y=False, name="Volume")
    fig.add_scatter(x=index, y=risk_rate, secondary_y=True, name="Risk Rate")
    if show_average:
        average_risk = marginal_table.NB_BAD_PAYER.sum() / marginal_table.NB_APPLICANTS.sum()
        fig.add_hline(y=average_risk, secondary_y=True, line_dash="dash", opacity=0.8)
    fig.update_yaxes(title_text="Risk Rate", secondary_y=True)
    fig.update_yaxes(title_text="Volume", secondary_y=False)
    fig["layout"]["yaxis2"]["showgrid"] = False
    fig.update_layout(title="Risk Rate by Modalitie")
    return fig
def default(data: pandas.DataFrame, h: int, r, nb_obs: str | None = None) -> pandas.Series:
    """Créer H{h}_R{r} (ou H{h}_CTX) à partir de HMIN_*_min; met NaN si nb_obs < h."""
    _r = "CTX" if r == "CTX" else (f"R{int(r)}" if int(r) > 0 else None)
    if _r is None:
        raise ValueError("Parameter r must be > 0 or 'CTX'.")
    hmin_col = f"HMIN_{_r}_min"
    if hmin_col not in data.columns:
        raise ValueError(f"Variable {hmin_col} not in data.")
    s = (data[hmin_col] <= h).astype(float)
    s.name = f"H{h}_{_r}"
    if nb_obs is not None:
        if nb_obs not in data.columns:
            raise ValueError(f"nb_obs column '{nb_obs}' not found in data.")
        mask = data[nb_obs].fillna(0) < h
        s[mask] = numpy.nan
    return s
def default_and_ctx(data: pandas.DataFrame, h: int, r: int, nb_obs: str | None = None) -> pandas.Series:
    """Créer H{h}_CTX_R{r} = OR(H{h}_R{r}, H{h}_CTX); met NaN si nb_obs < h."""
    if int(r) <= 0:
        raise ValueError("Parameter r must be > 0.")
    hr = default(data, h, r, nb_obs=None)
    hctx = default(data, h, "CTX", nb_obs=None)
    out = numpy.logical_or(hr.fillna(0).astype(bool), hctx.fillna(0).astype(bool)).astype(float)
    out = pandas.Series(out, index=data.index, name=f"H{h}_CTX_R{int(r)}")
    if nb_obs is not None:
        if nb_obs not in data.columns:
            raise ValueError(f"nb_obs column '{nb_obs}' not found in data.")
        mask = data[nb_obs].fillna(0) < h
        out[mask] = numpy.nan
    return out
def defaults_table(data: pandas.DataFrame, hmax: int, rmax: int, ctx: bool = False, nb_obs: str | None = None, add_to_data: bool = True) -> pandas.DataFrame:
    """Ajoute toutes les variables Hh_Rj, Hh_CTX et Hh_CTX_Rj pour h<=hmax, j<=rmax."""
    if hmax <= 0 or rmax <= 0:
        raise ValueError("hmax and rmax must be positive integers.")
    frames: list[pandas.DataFrame | pandas.Series] = [data] if add_to_data else []
    for h in range(1, hmax + 1):
        if ctx:
            frames.append(default(data, h, "CTX", nb_obs))
        for r in range(1, rmax + 1):
            frames.append(default(data, h, r, nb_obs))
            if ctx:
                frames.append(default_and_ctx(data, h, r, nb_obs))
    return pandas.concat(frames, axis=1)
def _is_default_variable(name: str) -> bool:
    """Vérifie les formats: Hh_Rj, Hh_CTX_Rj, Hh_CTX."""
    pat = "^H[0-9]{1,2}_R[0-9]{1,2}$"
    pat2 = "^H[0-9]{1,2}_CTX_R[0-9]{1,2}$"
    pat3 = "^H[0-9]{1,2}_CTX$"
    return bool(re.match("|".join([pat, pat2, pat3]), str(name)))
def _default_sort_key(name: str) -> list:
    """Retourne [R, H] pour trier par arriéré puis horizon."""
    if not _is_default_variable(name):
        raise ValueError(f"Variable {name} does not have a default variable format.")
    parts = [int(x) for x in re.split(r"(\d+)", str(name)) if x.isdigit()]
    if len(parts) == 1:
        parts.append(float("inf"))
    return parts[::-1]
def select_and_sort_default_variables(variables: list) -> list:
    """Liste triée des variables défauts (CTX à la fin)."""
    _list = [v for v in variables if _is_default_variable(v)]
    _list_ctx = [v for v in _list if "CTX" in str(v)]
    _list = [v for v in _list if "CTX" not in str(v)]
    _list.sort(key=_default_sort_key)
    _list_ctx.sort(key=_default_sort_key)
    return _list + _list_ctx
def risk_tables(data: pandas.DataFrame, generation: str, rates: bool = False) -> dict:
    """Dictionnaire de tables de risque par critère {'Rj' ou 'CTX_Rj': DataFrame}."""
    defaults = select_and_sort_default_variables(list(data.columns))
    if len(defaults) == 0:
        raise ValueError("No default variables in data ('Hi_Rj' and/or 'Hi_CTX_Rj').")
    if generation not in data:
        raise ValueError(f"There is no variable {generation} in the dataset.")
    defaults_table_df = data[defaults].fillna(0)
    gp = defaults_table_df.groupby(data[generation])
    risk_table = gp.sum()
    total_series = gp.size()
    _s1 = {var[4:] for var in risk_table.columns if re.match(r"^H\d{2}_", var)}
    _s2 = {var[3:] for var in risk_table.columns if re.match(r"^H\d{1}_", var)}
    default_criteria = sorted(list(_s1.union(_s2)), key=lambda t: t.replace("C", "Z"))
    dictionary = {}
    for crit in default_criteria:
        pattern = re.compile(rf"^H\d{{1,2}}_{re.escape(crit)}$")
        vars_crit = select_and_sort_default_variables([v for v in risk_table if pattern.match(v)])
        partial = risk_table[vars_crit]
        if not rates:
            partial.insert(0, "Total", total_series)
        else:
            partial = partial.apply(lambda s: s / total_series)
        dictionary[crit] = partial
    return dictionary
def compute_generation(data: pandas.DataFrame, gen: str, horizon: str, period: str, h_min: int, h_max: int, h_step: int, figsize: float) -> pandas.DataFrame:
    """Taux d’activation par génération (axe x = génération)."""
    df_gen = data.groupby(data[gen].dt.to_period("M")).size()
    df_gen_da = data.groupby([data[gen].dt.to_period("M"), horizon]).size().to_frame().reset_index()
    df_gen_da = df_gen_da.rename(columns={0: "Size"})
    df_ar = pandas.DataFrame(columns=[gen, "H", "AR"])
    a = 0
    for i in list(df_gen_da[gen].unique()):
        for h in range(h_min, h_max, h_step):
            csum = df_gen_da.loc[(df_gen_da[gen] == i) & (df_gen_da[horizon] <= h), "Size"].sum()
            df_ar.loc[a, "AR"] = csum / df_gen[i]
            df_ar.loc[a, gen] = i
            df_ar.loc[a, "H"] = h
            a += 1
    df_ar[gen] = df_ar[gen].astype(str)
    df_ar[period] = df_ar[gen]
    return df_ar
def compute_horizon(data: pandas.DataFrame, gen: str, horizon: str, period: str, h_min: int, h_max: int, h_step: int, figsize: float) -> pandas.DataFrame:
    """Taux d’activation par horizon (axe x = horizon)."""
    df_gen = data.groupby(data[gen].dt.to_period(period)).size()
    df_gen_da = data.groupby([data[gen].dt.to_period(period), horizon]).size().to_frame().reset_index()
    df_gen_da = df_gen_da.loc[df_gen_da[horizon] <= h_max].rename(columns={0: "Size"})
    df_ar = pandas.DataFrame(columns=[period, "H", "AR"])
    a = 0
    for i in list(df_gen_da[gen].unique()):
        for h in range(h_min, h_max, 1):
            csum = df_gen_da.loc[(df_gen_da[gen] == i) & (df_gen_da[horizon] <= h), "Size"].sum()
            df_ar.loc[a, "AR"] = csum / df_gen[i]
            df_ar.loc[a, period] = i
            df_ar.loc[a, "H"] = h
            a += 1
    return df_ar
def plot_gen_horizon(data, gen, horizon, x_axis="g", h_min=0, h_max=19, h_step=3, period="Y", figsize=(550, 1000), save_path=None):
    """Figure des taux d’activation par génération/horizon."""
    if horizon not in data.columns:
        raise ValueError(horizon + " should be present in data frame")
    if gen not in data.columns:
        raise ValueError(gen + " should be present in data frame")
    if x_axis not in ["h", "g"]:
        raise ValueError("x_axis should be either g or h")
    if not isinstance(h_min, int):
        raise ValueError("h_min should be positive integer")
    if not isinstance(h_max, int) or h_max < 0 or h_max < h_min:
        raise ValueError("h_max should be a positive integer and >= h_min")
    df_ar = compute_generation(data, gen, horizon, period, h_min, h_max, h_step, figsize) if x_axis == "g" else compute_horizon(data, gen, horizon, period, h_min, h_max, h_step, figsize)
    fig = px.line(df_ar, x="H", y="AR", color=period, labels={"H": "Horizon", "AR": "Activation Rate", period: gen}, markers=True, width=figsize[1], height=figsize[0])
    return fig
def risk_curve(criterion: pandas.DataFrame, curve_type: str = "convergence", figsize: Tuple[int, int] = (550, 1000)):
    """Courbe de convergence/stabilité du risque."""
    plot_data = criterion.copy()
    plot_data.columns = [get_horizon(c)[1:] for c in plot_data]
    if curve_type == "stability":
        fig = px.line(plot_data, width=figsize[1], height=figsize[0], labels={"value": "Risk Rate"})
        fig.update_layout(legend_title_text="Horizon")
    elif curve_type == "convergence":
        fig = px.line(plot_data.T, labels={"index": "Horizon", "value": "Risk Rate"}, width=figsize[1], height=figsize[0])
        fig.update_layout(legend_title_text="Generation")
    else:
        raise ValueError("curve_type must be 'convergence' or 'stability'")
    return fig
def plot_horizon_activ(data: pandas.DataFrame, horizon: str, t: float = 0.7, include_inactive: bool = False, figsize: tuple = (550, 1000), save_path: str = None):
    """Cumul du taux d’activation par horizon et mois de franchissement du seuil t."""
    if horizon not in data.columns:
        raise ValueError(horizon + " must be in DataFrame")
    counts = pandas.Series(numpy.zeros(int(data[horizon].max()) + 1))
    counts[data[horizon].value_counts().sort_index().index] = data[horizon].value_counts().sort_index()
    counts = counts.cumsum() / (len(data) if include_inactive else counts.sum())
    m = counts[counts >= t].index[0]
    fig = px.bar(counts, labels={"value": "Cumulative Activation Rate", "index": "Horizon"}, width=figsize[1], height=figsize[0])
    fig.add_hline(y=t, line_dash="dash", line_color="red")
    fig.update_layout(showlegend=False)
    logger.info(f"Number of months to reach a cumulative rate of {t} is {round(m)}")
    return fig
def plot_evolution_risk_rates(data: pandas.DataFrame, date_column: str, ctx: bool = True, figsize: Tuple[int, int] = (550, 1000)):
    """Évolution des taux de risque par horizon pour les stades d’impayé."""
    if not isinstance(data, pandas.DataFrame):
        raise TypeError("Inserted data should be a pandas DataFrame, with Hi_(CTX)_Rj variables")
    _dict = risk_tables(data, date_column, rates=False)
    arrear_keys = sorted([k for k in _dict.keys() if re.match(r"^R\d+$", k)], key=lambda s: int(s[1:]))
    if not arrear_keys:
        raise ValueError("No arrears keys found in risk tables (expected 'R1', 'R2', ...).")
    base = _dict[arrear_keys[0]].sum(axis=0)
    hcols = [c for c in base.index if c.startswith("H")]
    horizons = sorted(set(get_horizon(c) for c in hcols), key=lambda s: int(s[1:]))
    arrears_per_horizon_df = pandas.DataFrame(index=horizons)
    for k in arrear_keys:
        s = _dict[k].sum(axis=0)
        s = s[[c for c in s.index if c.startswith("H")]]
        s.index = [get_horizon(c) for c in s.index]
        arrears_per_horizon_df[k] = s.groupby(level=0).sum().reindex(horizons).fillna(0).values
    if ctx and "CTX" in _dict:
        sctx = _dict["CTX"].sum(axis=0)
        hctx = [c for c in sctx.index if re.match(r"^H\d{1,2}_CTX$", c)]
        if hctx:
            sctx = sctx[hctx]
            sctx.index = [get_horizon(c) for c in sctx.index]
            arrears_per_horizon_df["CTX"] = sctx.groupby(level=0).sum().reindex(horizons).fillna(0).values
    perct_arrears_per_horizon_df = arrears_per_horizon_df.divide(arrears_per_horizon_df.sum(axis=1).replace(0, numpy.nan), axis=0) * 100
    fig = px.line(perct_arrears_per_horizon_df, x=perct_arrears_per_horizon_df.index, y=perct_arrears_per_horizon_df.columns, labels={"index": "Horizon", "value": "Risk Rate %"}, title="Risk Rate Evolution", width=figsize[1], height=figsize[0])
    fig.update_layout(legend_title_text="Criterion")
    return fig
def get_horizon(col_name: str) -> str:
    """Extrait 'H<number>' d’un nom de colonne, sinon renvoie le nom original."""
    m = re.search(r"(H)(\d+)", str(col_name))
    return f"H{m.group(2)}" if m else str(col_name)


