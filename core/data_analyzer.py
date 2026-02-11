"""Data analysis functions for uploaded datasets.

This module provides functions to load, summarize, and analyze
epidemiological datasets from CSV, Excel, or pasted text.
"""

import io

import pandas as pd


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
        Dict with n, mean, median, sd, q1, q3, iqr, min, max.
    """
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return {
            "n": 0, "mean": None, "median": None, "sd": None,
            "q1": None, "q3": None, "iqr": None, "min": None, "max": None,
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))

    return {
        "n": n,
        "mean": round(float(clean.mean()), 4),
        "median": round(float(clean.median()), 4),
        "sd": round(float(clean.std()), 4) if n > 1 else 0.0,
        "q1": round(q1, 4),
        "q3": round(q3, 4),
        "iqr": round(q3 - q1, 4),
        "min": round(float(clean.min()), 4),
        "max": round(float(clean.max()), 4),
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
