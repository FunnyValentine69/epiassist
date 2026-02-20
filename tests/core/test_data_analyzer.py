"""Tests for core/data_analyzer.py."""

import io

import numpy as np
import pandas as pd
import pytest

from core.data_analyzer import (
    build_contingency_table,
    descriptive_stats_categorical,
    descriptive_stats_numeric,
    grouped_descriptive_stats,
    grouped_weighted_descriptive_stats,
    load_data,
    summarize_columns,
    weighted_stats_categorical,
    weighted_stats_numeric,
)


# --- Test fixture: sample epi dataset ---

def _make_sample_df(n: int = 700) -> pd.DataFrame:
    """Create a sample DataFrame mimicking epi data (hearing loss study)."""
    rng = np.random.default_rng(42)
    hearing_loss = rng.choice(["Yes", "No"], size=n, p=[0.33, 0.67])
    unemployed = np.where(
        hearing_loss == "Yes",
        rng.choice(["Yes", "No"], size=n, p=[0.35, 0.65]),
        rng.choice(["Yes", "No"], size=n, p=[0.15, 0.85]),
    )
    age = rng.normal(50, 15, size=n).round(1)
    education = rng.choice(
        ["High School", "College", "Graduate"], size=n, p=[0.4, 0.4, 0.2]
    )
    df = pd.DataFrame({
        "hearing_loss": hearing_loss,
        "unemployed": unemployed,
        "age": age,
        "education": education,
    })
    return df


SAMPLE_DF = _make_sample_df()


def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")


def _df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to Excel bytes."""
    buf = io.BytesIO()
    df.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


# ============================================================
# TestLoadData
# ============================================================

class TestLoadData:
    """Tests for load_data function."""

    def test_load_csv(self):
        csv_bytes = _df_to_csv_bytes(SAMPLE_DF)
        df = load_data(csv_bytes, "csv")
        assert len(df) == 700
        assert list(df.columns) == ["hearing_loss", "unemployed", "age", "education"]

    def test_load_excel(self):
        excel_bytes = _df_to_excel_bytes(SAMPLE_DF)
        df = load_data(excel_bytes, "excel")
        assert len(df) == 700
        assert "hearing_loss" in df.columns

    def test_load_paste_tab_separated(self):
        text = "col1\tcol2\n1\ta\n2\tb\n3\tc"
        df = load_data(text, "paste")
        assert len(df) == 3
        assert list(df.columns) == ["col1", "col2"]

    def test_load_paste_comma_separated(self):
        text = "col1,col2\n1,a\n2,b"
        df = load_data(text, "paste")
        assert len(df) == 2

    def test_strips_column_whitespace(self):
        text = " name , age \nAlice,30\nBob,25"
        df = load_data(text, "paste")
        assert list(df.columns) == ["name", "age"]

    def test_empty_data_raises(self):
        empty_csv = b"col1,col2\n"
        with pytest.raises(ValueError, match="empty"):
            load_data(empty_csv, "csv")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported format"):
            load_data(b"data", "json")


# ============================================================
# TestSummarizeColumns
# ============================================================

class TestSummarizeColumns:
    """Tests for summarize_columns function."""

    def test_detects_numeric_column(self):
        summaries = summarize_columns(SAMPLE_DF)
        age_summary = next(s for s in summaries if s["column"] == "age")
        assert age_summary["type"] == "numeric"

    def test_detects_categorical_column(self):
        summaries = summarize_columns(SAMPLE_DF)
        edu_summary = next(s for s in summaries if s["column"] == "education")
        assert edu_summary["type"] == "categorical"

    def test_binary_numeric_is_categorical(self):
        df = pd.DataFrame({"binary": [0, 1, 0, 1, 0, 1, 0, 1]})
        summaries = summarize_columns(df)
        assert summaries[0]["type"] == "categorical"

    def test_missing_values_counted(self):
        df = pd.DataFrame({"x": [1, 2, None, 4, None]})
        summaries = summarize_columns(df)
        assert summaries[0]["n_missing"] == 2
        assert summaries[0]["pct_missing"] == 40.0

    def test_samples_limited(self):
        df = pd.DataFrame({"x": range(100)})
        summaries = summarize_columns(df)
        assert len(summaries[0]["samples"]) <= 5


# ============================================================
# TestDescriptiveStatsNumeric
# ============================================================

class TestDescriptiveStatsNumeric:
    """Tests for descriptive_stats_numeric function."""

    def test_known_values(self):
        series = pd.Series([2, 4, 6, 8, 10])
        stats = descriptive_stats_numeric(series)
        assert stats["n"] == 5
        assert stats["mean"] == 6.0
        assert stats["median"] == 6.0
        assert stats["min"] == 2.0
        assert stats["max"] == 10.0

    def test_nan_handling(self):
        series = pd.Series([1, 2, None, 4, None])
        stats = descriptive_stats_numeric(series)
        assert stats["n"] == 3

    def test_single_value(self):
        series = pd.Series([5.0])
        stats = descriptive_stats_numeric(series)
        assert stats["n"] == 1
        assert stats["mean"] == 5.0
        assert stats["sd"] == 0.0

    def test_empty_series(self):
        series = pd.Series([], dtype=float)
        stats = descriptive_stats_numeric(series)
        assert stats["n"] == 0
        assert stats["mean"] is None


# ============================================================
# TestDescriptiveStatsNumericEnhanced
# ============================================================

class TestDescriptiveStatsNumericEnhanced:
    """Tests for the enhanced fields in descriptive_stats_numeric."""

    def test_variance_equals_sd_squared(self):
        series = pd.Series([2, 4, 6, 8, 10])
        stats = descriptive_stats_numeric(series)
        assert abs(stats["variance"] - stats["sd"] ** 2) < 0.001

    def test_iqr_present(self):
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        stats = descriptive_stats_numeric(series)
        assert stats["iqr"] == round(stats["q3"] - stats["q1"], 4)

    def test_skewness_symmetric(self):
        """A symmetric distribution should have skewness near zero."""
        series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
        stats = descriptive_stats_numeric(series)
        assert abs(stats["skewness"]) < 0.5

    def test_kurtosis_normal_like(self):
        """Normal-like data should have excess kurtosis near zero."""
        np.random.seed(42)
        series = pd.Series(np.random.normal(0, 1, 1000))
        stats = descriptive_stats_numeric(series)
        assert abs(stats["kurtosis"]) < 1.0

    def test_mode_returns_most_frequent(self):
        series = pd.Series([1, 2, 2, 3, 3, 3])
        stats = descriptive_stats_numeric(series)
        assert stats["mode"] == 3.0

    def test_missing_count_and_pct(self):
        series = pd.Series([1, 2, None, 4, None])
        stats = descriptive_stats_numeric(series)
        assert stats["n_missing"] == 2
        assert stats["missing_pct"] == 40.0

    def test_ci_for_mean(self):
        """95% CI should contain the sample mean."""
        series = pd.Series([10, 20, 30, 40, 50])
        stats = descriptive_stats_numeric(series)
        assert stats["ci_lower"] is not None
        assert stats["ci_upper"] is not None
        assert stats["ci_lower"] < stats["mean"] < stats["ci_upper"]

    def test_ci_none_for_single_value(self):
        series = pd.Series([5.0])
        stats = descriptive_stats_numeric(series)
        assert stats["ci_lower"] is None
        assert stats["ci_upper"] is None

    def test_skewness_none_for_two_values(self):
        series = pd.Series([1, 2])
        stats = descriptive_stats_numeric(series)
        assert stats["skewness"] is None

    def test_kurtosis_none_for_three_values(self):
        series = pd.Series([1, 2, 3])
        stats = descriptive_stats_numeric(series)
        assert stats["kurtosis"] is None

    def test_zero_variance_ci_equals_mean(self):
        """Zero-variance data should have CI = [mean, mean]."""
        series = pd.Series([5.0, 5.0, 5.0])
        stats = descriptive_stats_numeric(series)
        assert stats["ci_lower"] == stats["mean"]
        assert stats["ci_upper"] == stats["mean"]

    def test_inf_values_filtered(self):
        """Inf values should be excluded, not poison stats."""
        series = pd.Series([1.0, float("inf"), 3.0, 4.0])
        stats = descriptive_stats_numeric(series)
        assert stats["n"] == 3  # inf excluded
        assert stats["mean"] is not None
        # Mean of [1, 3, 4] is not inf
        assert stats["mean"] < 100

    def test_empty_returns_all_none(self):
        series = pd.Series([], dtype=float)
        stats = descriptive_stats_numeric(series)
        assert stats["variance"] is None
        assert stats["skewness"] is None
        assert stats["kurtosis"] is None
        assert stats["mode"] is None
        assert stats["ci_lower"] is None


# ============================================================
# TestWeightedStatsNumericEnhanced
# ============================================================

class TestWeightedStatsNumericEnhanced:
    """Tests for enhanced fields in weighted_stats_numeric."""

    def test_variance_equals_sd_squared(self):
        series = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        assert abs(stats["variance"] - stats["sd"] ** 2) < 0.001

    def test_missing_pct(self):
        series = pd.Series([1.0, 2.0, np.nan, 4.0])
        weights = pd.Series([1.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        assert stats["n_missing"] == 1
        assert stats["missing_pct"] == 25.0

    def test_ci_present(self):
        series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        assert stats["ci_lower"] is not None
        assert stats["ci_lower"] < stats["mean"] < stats["ci_upper"]

    def test_mode_present(self):
        series = pd.Series([1.0, 2.0, 2.0, 3.0])
        weights = pd.Series([1.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        assert stats["mode"] == 2.0

    def test_skewness_present(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        assert stats["skewness"] is not None


# ============================================================
# TestDescriptiveStatsCategorical
# ============================================================

class TestDescriptiveStatsCategorical:
    """Tests for descriptive_stats_categorical function."""

    def test_frequencies(self):
        series = pd.Series(["A", "B", "A", "A", "B"])
        stats = descriptive_stats_categorical(series)
        assert stats["n"] == 5
        assert stats["n_missing"] == 0
        assert len(stats["categories"]) == 2

    def test_sort_order(self):
        series = pd.Series(["A", "B", "B", "C", "C", "C"])
        stats = descriptive_stats_categorical(series)
        # First category should be most frequent
        assert stats["categories"][0]["value"] == "C"
        assert stats["categories"][0]["count"] == 3

    def test_nan_counted_separately(self):
        series = pd.Series(["A", "B", None, "A", None])
        stats = descriptive_stats_categorical(series)
        assert stats["n_missing"] == 2
        assert stats["n"] == 5

    def test_proportions_sum_to_one(self):
        series = pd.Series(["X", "Y", "Z", "X", "Y", "Z", "X"])
        stats = descriptive_stats_categorical(series)
        total_prop = sum(c["proportion"] for c in stats["categories"])
        assert abs(total_prop - 1.0) < 0.01


# ============================================================
# TestGroupedDescriptiveStats
# ============================================================

class TestGroupedDescriptiveStats:
    """Tests for grouped_descriptive_stats function."""

    def test_numeric_grouped(self):
        result = grouped_descriptive_stats(SAMPLE_DF, "age", "hearing_loss")
        assert "Yes" in result
        assert "No" in result
        assert "n" in result["Yes"]
        assert "mean" in result["Yes"]

    def test_categorical_grouped(self):
        result = grouped_descriptive_stats(SAMPLE_DF, "education", "hearing_loss")
        assert "Yes" in result
        assert "categories" in result["Yes"]

    def test_group_keys_match_unique(self):
        result = grouped_descriptive_stats(SAMPLE_DF, "age", "hearing_loss")
        expected_keys = set(SAMPLE_DF["hearing_loss"].unique())
        assert set(result.keys()) == expected_keys


# ============================================================
# TestWeightedStatsNumeric
# ============================================================

class TestWeightedStatsNumeric:
    """Tests for weighted_stats_numeric function."""

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights should give approximately the same mean as unweighted."""
        series = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        assert stats["n"] == 5
        assert abs(stats["mean"] - 6.0) < 0.01

    def test_heavy_weight_shifts_mean(self):
        """A heavy weight on one observation shifts the mean toward it."""
        series = pd.Series([1.0, 10.0])
        weights = pd.Series([1.0, 100.0])
        stats = weighted_stats_numeric(series, weights)
        # Mean should be much closer to 10 than 1
        assert stats["mean"] > 9.0

    def test_nan_excluded(self):
        """NaN in series or weights is excluded."""
        series = pd.Series([1.0, 2.0, np.nan, 4.0])
        weights = pd.Series([1.0, np.nan, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        # Only rows where both are non-NaN: rows 0 and 3
        assert stats["n"] == 2

    def test_empty_series_returns_zero(self):
        """Empty series returns n=0."""
        series = pd.Series([], dtype=float)
        weights = pd.Series([], dtype=float)
        stats = weighted_stats_numeric(series, weights)
        assert stats["n"] == 0
        assert stats["mean"] is None

    def test_effective_n_computed(self):
        """Effective N uses Kish's formula."""
        series = pd.Series([1.0, 2.0, 3.0])
        weights = pd.Series([1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        # Uniform weights: effective_n = (3)^2 / 3 = 3.0
        assert abs(stats["effective_n"] - 3.0) < 0.1

    def test_single_observation(self):
        """Single observation returns n=1, sd=0.0, correct mean."""
        series = pd.Series([7.5])
        weights = pd.Series([2.0])
        stats = weighted_stats_numeric(series, weights)
        assert stats["n"] == 1
        assert stats["mean"] == 7.5
        assert stats["sd"] == 0.0
        assert stats["effective_n"] == 1.0

    def test_effective_n_unequal_weights(self):
        """Highly unequal weights produce effective_n much less than n."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        weights = pd.Series([100.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_numeric(series, weights)
        # Kish: (103)^2 / (10000+1+1+1) ~ 1.06
        assert stats["effective_n"] < 2.0
        assert stats["n"] == 4


# ============================================================
# TestWeightedStatsCategorical
# ============================================================

class TestWeightedStatsCategorical:
    """Tests for weighted_stats_categorical function."""

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights produce proportions matching unweighted."""
        series = pd.Series(["A", "B", "A", "A", "B"])
        weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        stats = weighted_stats_categorical(series, weights)
        # A: 3/5 = 0.6, B: 2/5 = 0.4
        a_cat = next(c for c in stats["categories"] if c["value"] == "A")
        assert abs(a_cat["proportion"] - 0.6) < 0.01

    def test_weighted_proportions_correct(self):
        """Weighted proportions computed correctly with known weights."""
        series = pd.Series(["A", "B"])
        weights = pd.Series([3.0, 1.0])
        stats = weighted_stats_categorical(series, weights)
        # A: 3/4 = 0.75, B: 1/4 = 0.25
        a_cat = next(c for c in stats["categories"] if c["value"] == "A")
        assert abs(a_cat["proportion"] - 0.75) < 0.01

    def test_nan_excluded(self):
        """NaN in series or weights is excluded from proportions."""
        series = pd.Series(["A", "B", None, "A", "B"])
        weights = pd.Series([1.0, np.nan, 1.0, 2.0, 1.0])
        stats = weighted_stats_categorical(series, weights)
        # Valid rows: 0 (A,1.0), 3 (A,2.0), 4 (B,1.0) → n_missing includes NaN from both
        assert stats["n_missing"] > 0
        total_prop = sum(c["proportion"] for c in stats["categories"])
        assert abs(total_prop - 1.0) < 0.01

    def test_proportions_sum_to_one(self):
        """Weighted proportions sum to 1.0 with multiple categories."""
        series = pd.Series(["X", "Y", "Z", "X", "Y", "Z", "X"])
        weights = pd.Series([2.0, 3.0, 1.0, 4.0, 1.0, 5.0, 2.0])
        stats = weighted_stats_categorical(series, weights)
        total_prop = sum(c["proportion"] for c in stats["categories"])
        assert abs(total_prop - 1.0) < 0.01


# ============================================================
# TestGroupedWeightedDescriptiveStats
# ============================================================

class TestGroupedWeightedDescriptiveStats:
    """Tests for grouped_weighted_descriptive_stats function."""

    def test_correct_group_keys(self):
        """Returns correct group keys from the grouping variable."""
        df = SAMPLE_DF.copy()
        df["weight"] = 1.0
        result = grouped_weighted_descriptive_stats(df, "age", "hearing_loss", "weight")
        assert "Yes" in result
        assert "No" in result

    def test_numeric_gets_weighted_stats(self):
        """Numeric variable gets weighted stats with effective_n."""
        df = SAMPLE_DF.copy()
        df["weight"] = np.random.default_rng(42).uniform(0.5, 5.0, size=len(df))
        result = grouped_weighted_descriptive_stats(df, "age", "hearing_loss", "weight")
        # Should have weighted numeric stats
        assert "mean" in result["Yes"]
        assert "effective_n" in result["Yes"]

    def test_categorical_gets_weighted_stats(self):
        """Categorical variable gets weighted categorical stats per group."""
        df = SAMPLE_DF.copy()
        df["weight"] = 1.0
        result = grouped_weighted_descriptive_stats(df, "education", "hearing_loss", "weight")
        assert "Yes" in result
        assert "categories" in result["Yes"]


# ============================================================
# TestBuildContingencyTable
# ============================================================

class TestBuildContingencyTable:
    """Tests for build_contingency_table function."""

    def test_known_values(self):
        df = pd.DataFrame({
            "outcome": ["Yes", "Yes", "No", "No", "Yes", "No"],
            "exposure": ["Yes", "No", "Yes", "No", "Yes", "No"],
        })
        result = build_contingency_table(df, "outcome", "exposure", "Yes", "Yes")
        # Exposed+Outcome+: rows 0, 4 → a=2
        # Exposed+Outcome-: row 2 → b=1
        # Unexposed+Outcome+: row 1 → c=1
        # Unexposed+Outcome-: rows 3, 5 → d=2
        assert result["a"] == 2
        assert result["b"] == 1
        assert result["c"] == 1
        assert result["d"] == 2

    def test_n_total_correct(self):
        result = build_contingency_table(
            SAMPLE_DF, "unemployed", "hearing_loss", "Yes", "Yes"
        )
        assert result["n_total"] == 700
        assert result["a"] + result["b"] + result["c"] + result["d"] == 700

    def test_nan_exclusion(self):
        df = pd.DataFrame({
            "outcome": ["Yes", None, "No", "Yes"],
            "exposure": ["Yes", "Yes", "No", None],
        })
        result = build_contingency_table(df, "outcome", "exposure", "Yes", "Yes")
        assert result["n_excluded"] == 2
        assert result["n_total"] == 2

    def test_multi_level_collapse(self):
        """Multi-level variables should collapse to binary (positive vs everything else)."""
        df = pd.DataFrame({
            "outcome": ["A", "B", "C", "A", "B", "C"],
            "exposure": ["X", "X", "Y", "Y", "X", "Y"],
        })
        result = build_contingency_table(df, "outcome", "exposure", "A", "X")
        # Exposed(X)+Outcome(A): row 0 → a=1
        # Exposed(X)+Outcome(!A): rows 1, 4 → b=2
        # Unexposed(!X)+Outcome(A): row 3 → c=1
        # Unexposed(!X)+Outcome(!A): rows 2, 5 → d=2
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"] == 1
        assert result["d"] == 2

    def test_bad_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            build_contingency_table(SAMPLE_DF, "nonexistent", "hearing_loss", "Yes", "Yes")

    def test_no_matches_raises(self):
        df = pd.DataFrame({
            "outcome": [None, None],
            "exposure": [None, None],
        })
        with pytest.raises(ValueError, match="No valid rows"):
            build_contingency_table(df, "outcome", "exposure", "Yes", "Yes")

    def test_numeric_positive_value(self):
        df = pd.DataFrame({
            "outcome": [1, 0, 1, 0, 1],
            "exposure": [1, 1, 0, 0, 1],
        })
        result = build_contingency_table(df, "outcome", "exposure", 1, 1)
        assert result["a"] == 2
        assert result["b"] == 1
        assert result["c"] == 1
        assert result["d"] == 1
