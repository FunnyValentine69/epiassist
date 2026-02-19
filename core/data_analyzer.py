"""Data analysis functions for uploaded datasets.

This module provides functions to load, summarize, and analyze
epidemiological datasets from CSV, Excel, or pasted text.
"""

import io
import math

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.weightstats import DescrStatsW


def _sanitize(val: float | None) -> float | None:
    """Replace NaN/inf with None to prevent silent garbage in output."""
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def load_data(source: bytes | str, format: str) -> pd.DataFrame:
    """Load a dataset from CSV bytes, Excel bytes, or pasted text.

    Args:
        source: File bytes (CSV/Excel) or string (pasted text).
        format: One of "csv", "excel", or "paste".

    Returns:
        Loaded DataFrame with stripped column names.

    Raises:
        ValueError: If the data is empty or format is unsupported.
    """
    if format == "csv":
        if not isinstance(source, bytes):
            raise ValueError("CSV format requires bytes input.")
        df = pd.read_csv(io.BytesIO(source))
    elif format == "excel":
        if not isinstance(source, bytes):
            raise ValueError("Excel format requires bytes input.")
        df = pd.read_excel(io.BytesIO(source), engine="openpyxl")
    elif format == "paste":
        if not isinstance(source, str):
            raise ValueError("Paste format requires string input.")
        df = pd.read_csv(io.StringIO(source), sep=None, engine="python")
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv', 'excel', or 'paste'.")

    if df.empty:
        raise ValueError("The loaded data is empty.")

    df.columns = df.columns.str.strip()
    return df


def summarize_columns(df: pd.DataFrame) -> list[dict]:
    """Summarize each column in a DataFrame.

    Args:
        df: The DataFrame to summarize.

    Returns:
        List of dicts with column name, detected type, missing count/percent,
        unique values, and sample values.
    """
    summaries = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_total = len(series)
        pct_missing = round(n_missing / n_total * 100, 1) if n_total > 0 else 0.0
        n_unique = int(series.nunique())

        col_type = "numeric" if pd.api.types.is_numeric_dtype(series) and n_unique > 10 else "categorical"
        samples = series.dropna().unique()[:5].tolist()

        summaries.append({
            "column": col,
            "type": col_type,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": n_unique,
            "samples": samples,
        })
    return summaries


def descriptive_stats_numeric(series: pd.Series) -> dict:
    """Calculate descriptive statistics for a numeric variable.

    Args:
        series: A pandas Series of numeric values.

    Returns:
        Dict with n, n_missing, missing_pct, mean, median, mode, sd, variance,
        q1, q3, iqr, min, max, skewness, kurtosis, ci_lower, ci_upper.
    """
    n_total = len(series)
    n_missing = int(series.isna().sum())
    clean = series.dropna()
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    missing_pct = round(n_missing / n_total * 100, 1) if n_total > 0 else 0.0

    if n == 0:
        return {
            "n": 0, "n_missing": n_missing, "missing_pct": missing_pct,
            "mean": None, "median": None, "mode": None,
            "sd": None, "variance": None,
            "q1": None, "q3": None, "iqr": None, "min": None, "max": None,
            "skewness": None, "kurtosis": None,
            "ci_lower": None, "ci_upper": None,
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    mean_val = float(clean.mean())
    sd_val = float(clean.std()) if n > 1 else 0.0

    # Mode: most frequent value (first if tied)
    mode_result = clean.mode()
    mode_val = float(mode_result.iloc[0]) if len(mode_result) > 0 else None

    # 95% CI for the mean (t-distribution)
    if n > 1 and sd_val > 0:
        se = sd_val / np.sqrt(n)
        t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
        ci_lower = round(mean_val - t_crit * se, 4)
        ci_upper = round(mean_val + t_crit * se, 4)
    elif n > 1:
        # Zero variance: CI collapses to the mean
        ci_lower = round(mean_val, 4)
        ci_upper = round(mean_val, 4)
    else:
        ci_lower = None
        ci_upper = None

    return {
        "n": n,
        "n_missing": n_missing,
        "missing_pct": missing_pct,
        "mean": _sanitize(round(mean_val, 4)),
        "median": _sanitize(round(float(clean.median()), 4)),
        "mode": _sanitize(round(mode_val, 4)) if mode_val is not None else None,
        "sd": _sanitize(round(sd_val, 4)),
        "variance": _sanitize(round(sd_val**2, 4)),
        "q1": _sanitize(round(q1, 4)),
        "q3": _sanitize(round(q3, 4)),
        "iqr": _sanitize(round(q3 - q1, 4)),
        "min": _sanitize(round(float(clean.min()), 4)),
        "max": _sanitize(round(float(clean.max()), 4)),
        "skewness": _sanitize(round(float(clean.skew()), 4)) if n > 2 else None,
        "kurtosis": _sanitize(round(float(clean.kurtosis()), 4)) if n > 3 else None,
        "ci_lower": _sanitize(ci_lower),
        "ci_upper": _sanitize(ci_upper),
    }


def descriptive_stats_categorical(series: pd.Series) -> dict:
    """Calculate descriptive statistics for a categorical variable.

    Args:
        series: A pandas Series of categorical values.

    Returns:
        Dict with n, n_missing, and categories list (value, count, proportion)
        sorted by count descending.
    """
    n_total = len(series)
    n_missing = int(series.isna().sum())
    n_valid = n_total - n_missing

    counts = series.dropna().value_counts()
    categories = [
        {
            "value": value,
            "count": int(count),
            "proportion": round(count / n_valid, 4) if n_valid > 0 else 0.0,
        }
        for value, count in counts.items()
    ]

    return {
        "n": n_total,
        "n_missing": n_missing,
        "categories": categories,
    }


def grouped_descriptive_stats(
    df: pd.DataFrame, variable: str, group_by: str
) -> dict:
    """Calculate descriptive statistics stratified by a grouping variable.

    Args:
        df: The DataFrame.
        variable: Column to describe.
        group_by: Column to group by.

    Returns:
        Dict mapping group values to descriptive stats dicts.
    """
    results = {}
    for group_val, group_df in df.groupby(group_by):
        series = group_df[variable]
        # Use same type heuristic as summarize_columns
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            results[group_val] = descriptive_stats_numeric(series)
        else:
            results[group_val] = descriptive_stats_categorical(series)
    return results


def _is_categorical(series: pd.Series) -> bool:
    """Check if a Series should be treated as categorical.

    Uses the same heuristic as summarize_columns:
    NOT numeric dtype OR nunique <= 10.

    Args:
        series: A pandas Series.

    Returns:
        True if categorical, False if numeric.
    """
    return not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 10


def weighted_stats_numeric(series: pd.Series, weights: pd.Series) -> dict:
    """Calculate weighted descriptive statistics for a numeric variable.

    Args:
        series: A pandas Series of numeric values.
        weights: A pandas Series of positive weights aligned with series.

    Returns:
        Dict with n, n_missing, missing_pct, mean, median, mode, sd, variance,
        q1, q3, iqr, min, max, skewness, kurtosis, ci_lower, ci_upper, effective_n.
    """
    n_total = len(series)
    n_missing = int(series.isna().sum())
    missing_pct = round(n_missing / n_total * 100, 1) if n_total > 0 else 0.0

    # Align and drop rows where either is NaN or infinite
    mask = series.notna() & weights.notna() & np.isfinite(series) & np.isfinite(weights)
    clean = series[mask]
    w = weights[mask]
    n = len(clean)

    if n == 0:
        return {
            "n": 0, "n_missing": n_missing, "missing_pct": missing_pct,
            "mean": None, "median": None, "mode": None,
            "sd": None, "variance": None,
            "q1": None, "q3": None, "iqr": None, "min": None, "max": None,
            "skewness": None, "kurtosis": None,
            "ci_lower": None, "ci_upper": None,
            "effective_n": None,
        }

    if (w <= 0).any():
        raise ValueError(
            f"Weights must be positive (> 0). Found {(w <= 0).sum()} non-positive weight(s)."
        )

    d = DescrStatsW(clean.values, weights=w.values)

    # Kish's effective sample size: (sum w)^2 / sum(w^2)
    sum_w = float(w.sum())
    sum_w2 = float((w**2).sum())
    effective_n = round(sum_w**2 / sum_w2, 1) if sum_w2 > 0 else 0.0

    q1 = float(d.quantile(0.25).iloc[0])
    q3 = float(d.quantile(0.75).iloc[0])
    mean_val = float(d.mean)
    sd_val = float(d.std) if n > 1 else 0.0

    # Mode: most frequent value (unweighted — mode is observation-level)
    mode_result = clean.mode()
    mode_val = float(mode_result.iloc[0]) if len(mode_result) > 0 else None

    # 95% CI for the weighted mean using effective N
    if n > 1 and sd_val > 0 and effective_n > 1:
        se = sd_val / np.sqrt(effective_n)
        t_crit = scipy_stats.t.ppf(0.975, df=max(1, int(effective_n) - 1))
        ci_lower = round(mean_val - t_crit * se, 4)
        ci_upper = round(mean_val + t_crit * se, 4)
    elif n > 1:
        # Zero variance: CI collapses to the mean
        ci_lower = round(mean_val, 4)
        ci_upper = round(mean_val, 4)
    else:
        ci_lower = None
        ci_upper = None

    return {
        "n": n,
        "n_missing": n_missing,
        "missing_pct": missing_pct,
        "mean": _sanitize(round(mean_val, 4)),
        "median": _sanitize(round(float(d.quantile(0.5).iloc[0]), 4)),
        "mode": _sanitize(round(mode_val, 4)) if mode_val is not None else None,
        "sd": _sanitize(round(sd_val, 4)),
        "variance": _sanitize(round(sd_val**2, 4)),
        "q1": _sanitize(round(q1, 4)),
        "q3": _sanitize(round(q3, 4)),
        "iqr": _sanitize(round(q3 - q1, 4)),
        # min/max are inherently unweighted (observation-level extremes)
        "min": _sanitize(round(float(clean.min()), 4)),
        "max": _sanitize(round(float(clean.max()), 4)),
        "skewness": _sanitize(round(float(clean.skew()), 4)) if n > 2 else None,
        "kurtosis": _sanitize(round(float(clean.kurtosis()), 4)) if n > 3 else None,
        "ci_lower": _sanitize(ci_lower),
        "ci_upper": _sanitize(ci_upper),
        "effective_n": effective_n,
    }


def weighted_stats_categorical(series: pd.Series, weights: pd.Series) -> dict:
    """Calculate weighted descriptive statistics for a categorical variable.

    Args:
        series: A pandas Series of categorical values.
        weights: A pandas Series of positive weights aligned with series.

    Returns:
        Dict with n, n_missing, and categories list (value, count, proportion)
        sorted by weighted count descending.
    """
    n_total = len(series)
    mask = series.notna() & weights.notna()
    clean = series[mask]
    w = weights[mask]
    n_missing = n_total - len(clean)

    if len(w) > 0 and (w <= 0).any():
        raise ValueError(
            f"Weights must be positive (> 0). Found {(w <= 0).sum()} non-positive weight(s)."
        )

    # Group by value, sum weights (sorted descending)
    weighted_counts = w.groupby(clean).sum().sort_values(ascending=False)
    total_weight = float(weighted_counts.sum())

    categories = [
        {
            "value": val,
            "count": round(float(wt), 2),
            "proportion": round(float(wt) / total_weight, 4) if total_weight > 0 else 0.0,
        }
        for val, wt in weighted_counts.items()
    ]

    return {
        "n": n_total,
        "n_missing": n_missing,
        "categories": categories,
    }


def grouped_weighted_descriptive_stats(
    df: pd.DataFrame, variable: str, group_by: str, weight_col: str
) -> dict:
    """Calculate weighted descriptive statistics stratified by a grouping variable.

    Args:
        df: The DataFrame.
        variable: Column to describe.
        group_by: Column to group by.
        weight_col: Column containing survey weights.

    Returns:
        Dict mapping group values to weighted descriptive stats dicts.

    Raises:
        ValueError: If any column name not found in DataFrame.
    """
    for col, label in [(variable, "Variable"), (group_by, "Group by"), (weight_col, "Weight")]:
        if col not in df.columns:
            raise ValueError(f"{label} column '{col}' not found in DataFrame.")

    results = {}
    for group_val, group_df in df.groupby(group_by):
        series = group_df[variable]
        w = group_df[weight_col]
        if _is_categorical(series):
            results[group_val] = weighted_stats_categorical(series, w)
        else:
            results[group_val] = weighted_stats_numeric(series, w)
    return results


def build_contingency_table(
    df: pd.DataFrame,
    outcome_col: str,
    exposure_col: str,
    outcome_positive: object,
    exposure_positive: object,
) -> dict:
    """Build a 2x2 contingency table from a DataFrame.

    Table layout:
                    Outcome+  Outcome-
        Exposed        a         b
        Unexposed      c         d

    Args:
        df: The DataFrame.
        outcome_col: Column name for outcome variable.
        exposure_col: Column name for exposure variable.
        outcome_positive: Value indicating positive outcome.
        exposure_positive: Value indicating positive exposure.

    Returns:
        Dict with a, b, c, d cell counts, n_excluded, and n_total.

    Raises:
        ValueError: If columns don't exist or no valid rows.
    """
    for col in [outcome_col, exposure_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # Drop rows with NaN in either column
    subset = df[[outcome_col, exposure_col]].dropna()
    n_excluded = len(df) - len(subset)

    if len(subset) == 0:
        raise ValueError("No valid rows after excluding missing values.")

    # Convert positive values to match column dtype if needed
    outcome_pos = _coerce_value(outcome_positive, subset[outcome_col])
    exposure_pos = _coerce_value(exposure_positive, subset[exposure_col])

    exposed = subset[exposure_col] == exposure_pos
    outcome = subset[outcome_col] == outcome_pos

    a = int((exposed & outcome).sum())
    b = int((exposed & ~outcome).sum())
    c = int((~exposed & outcome).sum())
    d = int((~exposed & ~outcome).sum())

    if a + b + c + d == 0:
        raise ValueError("No observations matched the specified positive values.")

    if a + b == 0:
        raise ValueError("No exposed observations found. Check the positive exposure value.")

    if c + d == 0:
        raise ValueError("No unexposed observations found. All rows match the exposure value.")

    return {
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "n_excluded": n_excluded,
        "n_total": len(subset),
    }


def _coerce_value(value: object, series: pd.Series) -> object:
    """Coerce a value to match the dtype of a Series.

    Args:
        value: The value to coerce.
        series: The Series whose dtype to match.

    Returns:
        The coerced value.
    """
    if pd.api.types.is_numeric_dtype(series):
        try:
            return type(series.dropna().iloc[0])(value)
        except (IndexError, ValueError, TypeError):
            return value
    return value
