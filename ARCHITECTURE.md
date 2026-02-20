# EpiAssist Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT UI                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐│
│  │   DAG    │ │  Stats   │ │Hypothesis│ │  Paper   │ │  Power   │ │  Meta-   │ │  Data    ││
│  │ Builder  │ │Calculator│ │ Testing  │ │ Analyzer │ │ Analysis │ │ Analysis │ │ Analysis ││
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘│
│       │            │            │            │            │            │            │      │
└───────┼────────────┼────────────┼────────────┼────────────┼────────────┼────────────┼──────┘
        │            │            │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CORE MODULES                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │   dag    │ │  stats   │ │confounder│ │  paper   │ │  power   │      │
│  │ engine   │ │calculator│ │ detector │ │  parser  │ │calculator│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│  ┌──────────┐                            ┌──────────┐ ┌──────────┐     │
│  ┌──────────┐ ┌──────────┐                ┌──────────┐ ┌──────────┐     │
│  │   meta   │ │  data    │                │ e_value  │ │   llm    │     │
│  │ analysis │ │ analyzer │                └──────────┘ │extractor │     │
│  └──────────┘ └──────────┘                             └──────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            UTILITIES                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐       │
│  │      interpretations.py     │  │       constants.py          │       │
│  │  (Plain English outputs)    │  │  (Reference values/limits)  │       │
│  └─────────────────────────────┘  └─────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                             SCRIPTS                                      │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  diagnose_extraction.py - Pattern debugging and extraction tests │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### app.py (Main Entry Point)

- Streamlit page configuration and theming
- Navigation sidebar setup
- Global session state initialization
- Welcome/landing page content

### pages/ (UI Components)

#### 1_DAG_Builder.py
- Interactive node/edge creation interface
- Variable type selection (exposure, outcome, confounder, mediator)
- DAG visualization with Graphviz
- Confounder detection trigger and display
- Export/import DAG functionality

#### 2_Stats_Calculator.py
- Tabbed layout: "2x2 Table (OR/RR/RD)", "SMR/SIR Calculator", "Direct Standardization"
- 2x2 contingency table input with OR, RR, RD, Chi-square
- SMR/SIR: simple mode (observed/expected) or stratified mode (person-time + reference rates)
- Direct standardization: age-adjusted rates with built-in standard populations (US 2000, WHO World, Segi World)
- Confidence interval display
- Natural language interpretations

#### 3_Hypothesis_Testing.py
- Research question formulation
- Null/alternative hypothesis generator
- Study design recommendations
- Bias checklist (selection, information, confounding)

#### 4_Paper_Analyzer.py
- PDF upload interface
- Extracted statistics display
- Confidence interval parsing
- P-value identification
- Study summary generation

#### 5_Power_Analysis.py
- Sample size calculator
- Power curve visualization
- E-value sensitivity analysis
- Effect size estimation tools

#### 6_Meta_Analysis.py
- Study data entry (manual or import from Paper Analyzer)
- Measure type and model selection
- Forest plot generation (Plotly)
- Funnel plot for publication bias assessment
- Pooled estimate display with heterogeneity statistics

#### 7_Data_Analysis.py
- Data upload (CSV, Excel) and paste interface, plus built-in demo dataset (synthetic NHANES)
- Variable role assignment with explanations (outcome, exposure, confounders, mediators, weights)
- Descriptive statistics with grouped comparisons and histograms
- Auto-generated 2x2 cross-tabulation using `build_contingency_table`
- Reuses `stats_calculator` for OR, RR, RD, Chi-square on derived table
- Mantel-Haenszel stratified analysis when confounders are assigned (adjusted OR/RR, Breslow-Day test)
- E-value sensitivity analysis auto-computed from crude OR (and MH-adjusted OR when available)
- Regression analysis tab: logistic (OR), linear (β), Poisson (IRR) via `core/regression.py`
- Survey-weighted analysis: optional weight column for weighted descriptive stats (Tab 3) and weighted regression (Tab 5) using `freq_weights`
- Propensity score analysis tab: IPTW-based causal inference with PS estimation, common support assessment, balance diagnostics (Love plot), bootstrap CIs, and E-value integration via `core/propensity_score.py`

#### 8_Export_Report.py
- Manuscript Methods section generator (template-based, no LLM)
- Calls `utils/methods_generator.py` with session state snapshot
- Markdown download and copy-to-clipboard for generated text
- PDF report export via `core/report_generator.py` (fpdf2, text + tables, no figures)

### core/ (Business Logic)

#### dag_engine.py
```python
class DAGEngine:
    def __init__(self) -> None
    def add_node(self, name: str, node_type: str) -> None
    def add_edge(self, source: str, target: str) -> None
    def remove_node(self, name: str) -> None
    def remove_edge(self, source: str, target: str) -> None
    def get_all_paths(self, source: str, target: str) -> list[list[str]]
    def render_graphviz(self) -> graphviz.Digraph
    def to_dict(self) -> dict
    def from_dict(self, data: dict) -> None
```

#### stats_calculator.py
```python
def calculate_odds_ratio(a: int, b: int, c: int, d: int) -> dict
def calculate_risk_ratio(a: int, b: int, c: int, d: int) -> dict
def calculate_risk_difference(a: int, b: int, c: int, d: int) -> dict
def calculate_chi_square(a: int, b: int, c: int, d: int) -> dict
def calculate_confidence_interval(estimate: float, se: float, level: float = 0.95) -> tuple[float, float]
def calculate_mantel_haenszel(strata: list[dict]) -> dict
    # MH-adjusted OR/RR via statsmodels StratifiedTable
    # Returns: or_value, or_ci, rr_value, rr_ci, mh_test, breslow_day, interpretation
```

#### confounder_detector.py
```python
def find_confounders(dag: nx.DiGraph, exposure: str, outcome: str) -> list[str]
def find_backdoor_paths(dag: nx.DiGraph, exposure: str, outcome: str) -> list[list[str]]
def suggest_adjustment_set(dag: nx.DiGraph, exposure: str, outcome: str) -> list[str]
def normalize_variable_name(name: str) -> str
    # Strips whitespace, lowercases, replaces _ and - with spaces
def match_columns_to_dag_nodes(column_names: list[str], dag_node_names: list[str]) -> dict[str, str]
    # Returns DAG node name → matched column name (exact normalized comparison)
def compare_adjustment_sets(dag_set: list[str], paper_set: list[str]) -> dict[str, list[str]]
    # Returns overlap, dag_only, paper_only (original names preserved)
```

#### paper_parser.py
```python
def extract_text_from_pdf(file_bytes: bytes) -> list[tuple[int, str]]
def find_effect_measures(text: str, page: int = 1) -> list[dict]
    # Returns dicts with: type (OR|HR|RR|PR|IRR|β), value, ci_lower, ci_upper,
    #                     adjusted (True|False|None), adjusted_for (str|None), context, page
def find_confidence_intervals(text: str, page: int = 1) -> list[dict]
def find_p_values(text: str, page: int = 1) -> list[dict]
def find_sample_sizes(text: str, page: int = 1) -> list[dict]
def find_beta_coefficients(text: str, page: int = 1) -> list[dict]
    # Returns dicts with: value, ci_lower, ci_upper, se, context, page
def find_mean_differences(text: str, page: int = 1) -> list[dict]
    # Returns dicts with: value, ci_lower, ci_upper, context, page
def find_standard_deviations(text: str, page: int = 1) -> list[dict]
    # Returns dicts with: mean, value, type (SD|SE), context, page
def find_weighted_statistics(text: str, page: int = 1) -> list[dict]
    # Returns dicts with: stat_type, value, weight_method (str|None), context, page
```

#### llm_extractor.py (provider-agnostic — delegates to llm_providers)
```python
def is_llm_available() -> tuple[bool, str | None]
    # Returns (available, provider_name) via detect_provider()
def extract_with_llm(text: str, page: int = 1) -> dict[str, list[dict]]
    # Delegates to best available provider, returns same 8-key dict as paper_results
def merge_results(regex_results: dict, llm_results: dict) -> dict[str, list[dict]]
    # Deduplicates using float-equal comparison, tags source="regex"|"llm"
```

#### llm_providers/ (LLM provider abstraction package)
```python
# __init__.py — Registry and auto-detection
def get_api_key(key_name: str) -> str | None
    # Checks st.secrets first, falls back to os.environ
def detect_provider() -> str | None
    # Priority: gemini (API key + SDK) > ollama (localhost) > None
def get_provider_functions(name: str) -> dict
    # Lazy-imports provider module, returns {"extract_stats": ..., "chat": ...}
def provider_display_name(name: str | None) -> str
    # "Gemini (cloud)" / "Ollama (local)" / "None"

# prompts.py — Shared extraction prompt and few-shot example
EXTRACTION_SYSTEM_PROMPT: str  # JSON-output system prompt for all providers
FEW_SHOT_USER: str             # Example input text
FEW_SHOT_ASSISTANT: str        # Example JSON output

# _parse.py — Shared JSON response parser
CATEGORIES: list[str]          # 8 category keys
def parse_extraction_response(raw_json: str, page: int) -> dict[str, list[dict]]
    # Defensive JSON parsing with per-category validators

# ollama.py — Local inference via Ollama HTTP API
def extract_stats(text: str, page: int) -> dict[str, list[dict]]
def chat(prompt: str) -> str | None

# gemini.py — Cloud inference via google-genai SDK
def extract_stats(text: str, page: int) -> dict[str, list[dict]]
def chat(prompt: str) -> str | None
```

#### power_calculator.py
```python
def calculate_sample_size(effect_size: float, alpha: float, power: float) -> int
def calculate_power(n: int, effect_size: float, alpha: float) -> float
def generate_power_curve(effect_size: float, alpha: float, n_range: tuple[int, int]) -> pd.DataFrame
```

#### meta_analysis.py
```python
def validate_studies(studies: list[dict]) -> list[str]
def _calculate_se_from_ci(ci_lower: float, ci_upper: float, is_log_scale: bool) -> float
def _prepare_studies(studies: list[dict], measure_type: str) -> list[dict]
def fixed_effect_meta(prepared: list[dict], measure_type: str) -> dict
def heterogeneity_stats(prepared: list[dict]) -> dict
def random_effects_meta(prepared: list[dict], tau_squared: float, measure_type: str) -> dict
def run_meta_analysis(studies: list[dict], measure_type: str, model: str = "both") -> dict
```

#### data_analyzer.py
```python
def load_data(source: bytes | str, format: str) -> pd.DataFrame
def summarize_columns(df: pd.DataFrame) -> list[dict]
def descriptive_stats_numeric(series: pd.Series) -> dict
    # Returns: n, n_missing, missing_pct, mean, median, mode, sd, variance,
    #   q1, q3, iqr, min, max, skewness, kurtosis, ci_lower, ci_upper
def descriptive_stats_categorical(series: pd.Series) -> dict
def grouped_descriptive_stats(df: pd.DataFrame, variable: str, group_by: str) -> dict
def weighted_stats_numeric(series: pd.Series, weights: pd.Series) -> dict
    # Weighted mean/sd/quantiles via DescrStatsW + Kish's effective_n
def weighted_stats_categorical(series: pd.Series, weights: pd.Series) -> dict
    # Weighted proportions (sum weights per category)
def grouped_weighted_descriptive_stats(df: pd.DataFrame, variable: str, group_by: str, weight_col: str) -> dict
    # Grouped analog of weighted_stats_numeric/categorical
def build_contingency_table(df: pd.DataFrame, outcome_col: str, exposure_col: str, outcome_positive: object, exposure_positive: object) -> dict
```

#### smr_calculator.py
```python
def calculate_smr(observed: int, expected: float, ci_level: float = 0.95) -> dict
    # SMR = observed / expected, exact Poisson CI via chi-squared
    # Returns: value, ci_lower, ci_upper, interpretation, observed, expected
def calculate_expected_events(strata: list[dict]) -> dict
    # Sums rate * person_time per stratum
    # Returns: expected, strata_details, total_person_time, total_observed
def calculate_smr_stratified(strata: list[dict], ci_level: float = 0.95) -> dict
    # Pipeline: calculate_expected_events → calculate_smr
    # Returns: SMR result + strata_details, total_person_time
```

#### direct_standardization.py
```python
def calculate_stratum_rates(strata: list[dict]) -> list[dict]
    # Computes rate = events/population and weighted_events = rate * standard_weight
def calculate_direct_standardized_rate(strata: list[dict], multiplier: int = 100_000, ci_level: float = 0.95) -> dict
    # Direct standardization with Fay-Feuer CI (gamma distribution)
    # Returns: value, ci_lower, ci_upper, interpretation, strata_details,
    #          total_standard_pop, total_events, total_population, crude_rate, multiplier
```

#### regression.py
```python
def run_logistic_regression(df, outcome_col, exposure_col, confounder_cols, outcome_positive, exposure_positive=None, weight_col=None) -> dict
def run_linear_regression(df, outcome_col, exposure_col, confounder_cols, exposure_positive=None, weight_col=None) -> dict
def run_poisson_regression(df, outcome_col, exposure_col, confounder_cols, exposure_positive=None, weight_col=None) -> dict
    # All use sm.GLM uniformly (freq_weights when weight_col provided).
    # Returns: model_type, weighted, n_observations, n_dropped, converged,
    # exposure_effect, coefficients, model_fit (AIC/BIC/R²), interpretation
```

#### propensity_score.py
```python
def estimate_propensity_scores(df, treatment_col, confounder_cols, treatment_positive, weight_col=None) -> dict
def assess_common_support(ps_treated, ps_control) -> dict
def calculate_iptw_weights(ps, treatment, estimand="ATE", stabilized=True, trim_quantile=0.0) -> dict
def calculate_smd(treated_values, control_values, treated_weights=None, control_weights=None) -> float
def balance_diagnostics(df, confounder_cols, treatment_binary, iptw_weights=None) -> dict
def estimate_treatment_effect(df, outcome_col, treatment_binary, iptw_weights, outcome_type="binary", ...) -> dict
def run_propensity_score_analysis(df, outcome_col, treatment_col, confounder_cols, ...) -> dict
    # Full IPTW pipeline: PS estimation → common support → weights → balance → treatment effect
    # Returns: ps_model, common_support, iptw, balance, treatment_effect, interpretation
```

#### mediation.py
```python
def fit_mediation_models(df, outcome_col, exposure_col, mediator_col, confounder_cols, exposure_positive, outcome_type="continuous", ...) -> dict
def calculate_mediation_effects(a, b, c, c_prime, se_a, se_b, outcome_type="continuous") -> dict
def bootstrap_mediation_ci(df, outcome_col, exposure_col, mediator_col, confounder_cols, exposure_positive, ..., n_boot=200) -> dict
def run_mediation_analysis(df, outcome_col, exposure_col, mediator_col, confounder_cols, exposure_positive, ...) -> dict
    # Baron-Kenny pipeline: 3 GLM models → effect decomposition → bootstrap CIs → interpretation
    # Returns: models, effects, ci, n_observations, n_dropped, n_boot, interpretation
```

#### e_value.py
```python
def calculate_e_value(point_estimate: float, ci_bound: float = None) -> dict
def interpret_e_value(e_value: float) -> str
```

#### report_generator.py
```python
class EpiAssistReport(FPDF):
    # Registers DejaVu Sans fonts (regular, bold, oblique) for Unicode support
def generate_report(session_state: dict) -> bytes
    # PDF report via fpdf2 with DejaVu Sans fonts (text + tables, no figures)
    # Sections: title page, data summary, DAG, effect estimates,
    #           regression, propensity score, mediation, meta-analysis,
    #           paper summary, methods section
    # Each section guarded by session state key presence
```

### utils/ (Shared Utilities)

#### interpretations.py
```python
def interpret_odds_ratio(or_value: float, ci_lower: float, ci_upper: float) -> str
def interpret_risk_ratio(rr_value: float, ci_lower: float, ci_upper: float) -> str
def interpret_p_value(p: float, alpha: float = 0.05) -> str
def interpret_power(power: float) -> str
def interpret_e_value(e_value: float) -> str
def interpret_heterogeneity(i_squared: float, q_p_value: float, num_studies: int) -> str
def interpret_direct_standardized_rate(adjusted_rate: float, ci_lower: float, ci_upper: float, crude_rate: float, multiplier: int) -> str
def interpret_smr(smr: float, ci_lower: float, ci_upper: float) -> str
def interpret_meta_analysis(pooled: float, ci_lower: float, ci_upper: float, measure_type: str, model: str) -> str
def interpret_mantel_haenszel(or_value: float, or_ci_lower: float, or_ci_upper: float, homogeneity_p: float | None, n_strata: int, confounder_name: str) -> str
def interpret_logistic_regression(exposure_name, or_value, ci_lower, ci_upper, p_value, confounder_names, n_obs, weighted=False) -> str
def interpret_linear_regression(exposure_name, beta, ci_lower, ci_upper, p_value, confounder_names, n_obs, r_squared, weighted=False) -> str
def interpret_poisson_regression(exposure_name, irr, ci_lower, ci_upper, p_value, confounder_names, n_obs, weighted=False) -> str
def interpret_propensity_score(estimand, effect_value, ci_lower, ci_upper, outcome_type, treatment_name, confounder_names, n_obs, effective_n, all_balanced, n_balanced, n_total_covariates, weighted=False) -> str
def interpret_mediation(mediator_name, exposure_name, outcome_name, indirect, direct, total, indirect_ci, direct_ci, sobel_p, proportion_mediated, method, n_obs, confounder_names, weighted=False) -> str
```

#### ui_helpers.py
```python
def styled_banner(text: str, level: str = "success") -> None
    # Dark-mode-compatible colored banner (success/warning/error/info)
def robustness_badge(e_value: float) -> None
    # E-value robustness display using styled_banner
def plot_download_button(fig: object, filename: str = "plot", label: str = "Download Plot (HTML)") -> None
    # Plotly figure download as interactive HTML (no kaleido needed)
```

#### methods_generator.py
```python
def generate_methods_section(state: dict) -> str
    # Template-based Methods section from session state
    # Sections: study design, descriptive, cross-tab, regression,
    #           propensity score, mediation, meta-analysis, sensitivity, software
    # Injects computed results (OR/RR, regression coefficients, pooled estimates, E-values)
```

#### constants.py
```python
ALPHA_DEFAULT = 0.05
POWER_DEFAULT = 0.80
CI_LEVEL_DEFAULT = 0.95
Z_SCORE_95 = 1.96

EFFECT_SIZE_THRESHOLDS = {
    "small": 0.2,
    "medium": 0.5,
    "large": 0.8
}

NODE_COLORS = {
    "exposure": "#FF6B6B",
    "outcome": "#4ECDC4",
    "confounder": "#FFE66D",
    "mediator": "#95E1D3"
}

I_SQUARED_THRESHOLDS = {"low": 25, "moderate": 50, "high": 75}
RATIO_MEASURES = {"OR", "RR", "HR", "PR", "IRR"}
DIFFERENCE_MEASURES = {"MD", "RD", "beta"}
META_MEASURE_LABELS = {"OR": "Odds Ratio", ...}
STANDARD_POPULATIONS = {"US 2000": [...], "WHO World": [...], "Segi World": [...]}
RATE_MULTIPLIERS = {"per 1,000": 1000, "per 10,000": 10000, "per 100,000": 100000}
SMD_BALANCE_THRESHOLD = 0.1  # Propensity score covariate balance (Austin 2009)
```

## Data Flow Diagrams

### DAG Builder Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ User Input  │───▶│ DAGEngine   │───▶│ NetworkX    │───▶│ Graphviz    │
│ (node/edge) │    │ .add_node() │    │ DiGraph     │    │ render      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   Confounder    │
                                    │   Detector      │
                                    └────────┬────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │ Adjustment Set  │
                                    │ Recommendation  │
                                    └─────────────────┘
```

### Statistics Calculator Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 2x2 Table   │───▶│   Stats     │───▶│ Interpret-  │───▶│  Display    │
│   Input     │    │ Calculator  │    │   ations    │    │  Results    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
     │                   │
     │              ┌────┴────┐
     │              ▼         ▼
     │         ┌────────┐ ┌────────┐
     │         │   OR   │ │   RR   │ ...
     │         └────────┘ └────────┘
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  Input:  | Disease+ | Disease- |    Output:                     │
│  ────────┼──────────┼──────────┤    - OR: 2.5 (1.8-3.4)        │
│  Exposed |    a     |    b     |    - RR: 1.9 (1.4-2.6)        │
│  Unexpo. |    c     |    d     |    - Chi-sq: 45.2, p<0.001    │
└─────────────────────────────────────────────────────────────────┘
```

### Paper Analyzer Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ PDF Upload  │───▶│  PyMuPDF    │───▶│   Regex     │───▶│ Regex Stats │
│             │    │  Extract    │    │  Patterns   │    │             │
└─────────────┘    └─────────────┘    └──────┬──────┘    └──────┬──────┘
                                             │                  │
                                             ▼                  │
                                    ┌─────────────────┐         │
                                    │  LLM Providers  │         │
                                    │  Gemini (cloud)  │         │
                                    │  Ollama (local)  │         │
                                    │  [auto-detect]  │         │
                                    └────────┬────────┘         │
                                             │                  │
                                             ▼                  ▼
                                    ┌─────────────────────────────┐
                                    │   Merge + Deduplicate       │
                                    │  (float-equal comparison)   │
                                    └──────────────┬──────────────┘
                                                   │
         ┌─────────────────────────────────────────┤
         ▼           ▼           ▼           ▼     ▼       ▼
   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ OR/HR/RR │ │  Beta    │ │   CIs    │ │ P-values │ │  SD/SE   │
   └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
```

### Power Analysis Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Effect Size │───▶│   Power     │───▶│   Plotly    │
│ Alpha/Power │    │ Calculator  │    │   Curves    │
└─────────────┘    └─────────────┘    └─────────────┘
                         │
                         ▼
                   ┌──────────────┐
                   │ Sample Size  │
                   │ Requirement  │
                   └──────────────┘
```

### Meta-Analysis Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Study Data   │───▶│  Validate   │───▶│ Prepare     │───▶│  Fixed /    │
│ (manual or   │    │  Studies    │    │ (log-scale  │    │  Random     │
│  imported)   │    │             │    │  for ratios)│    │  Effects    │
└─────────────┘    └─────────────┘    └──────┬──────┘    └──────┬──────┘
                                             │                  │
                                             ▼                  ▼
                                    ┌─────────────────┐ ┌─────────────────┐
                                    │  Heterogeneity  │ │  Forest Plot    │
                                    │  (Q, I², τ²)    │ │  Funnel Plot    │
                                    └─────────────────┘ └─────────────────┘
```

### Data Analysis Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ CSV/Excel/  │───▶│  load_data  │───▶│ summarize   │───▶│  Variable   │
│ Paste Input │    │             │    │ _columns    │    │  Roles UI   │
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                        ┌───────────────────────────────────────┤
                        ▼                                       ▼
               ┌─────────────────┐                     ┌─────────────────┐
               │  Descriptive    │                     │ build_          │
               │  Stats (grouped │                     │ contingency_    │
               │  + histograms)  │                     │ table → a,b,c,d │
               └─────────────────┘                     └────────┬────────┘
                                                                │
                                                                ▼
                                                       ┌─────────────────┐
                                                       │ stats_calculator│
                                                       │ OR, RR, RD, χ² │
                                                       └────────┬────────┘
                                                                │
                                                                ▼
                                                       ┌─────────────────┐
                                                       │ Mantel-Haenszel │
                                                       │ (if confounders │
                                                       │  assigned)      │
                                                       └────────┬────────┘
                                                                │
                                                                ▼
                                                       ┌─────────────────┐
                                                       │ E-Value         │
                                                       │ (crude OR +     │
                                                       │  adjusted OR)   │
                                                       └─────────────────┘

               ┌───────────────────────────────────────┤
               ▼                                       │
      ┌─────────────────┐                              │
      │  Regression     │                              │
      │  (logistic/     │ ◄────────── exposure + confounders
      │  linear/Poisson)│
      └─────────────────┘
```

## Session State Schema

```python
st.session_state = {
    # DAG Builder
    "dag_graph": nx.DiGraph(),           # The DAG itself
    "dag_exposure": str | None,          # Selected exposure variable
    "dag_outcome": str | None,           # Selected outcome variable
    "dag_confounders": list[str],        # Detected confounders

    # Statistics Calculator
    "stats_table": {                      # Current 2x2 table
        "a": int, "b": int,
        "c": int, "d": int
    },
    "stats_results": dict | None,        # Last calculation results

    # Hypothesis Testing
    "hypo_research_question": str,
    "hypo_null": str,
    "hypo_alternative": str,
    "hypo_study_design": str,

    # Paper Analyzer
    "paper_text": str | None,            # Extracted PDF text
    "paper_stats": list[dict],           # Found statistics

    # Power Analysis
    "power_params": {
        "effect_size": float,
        "alpha": float,
        "power": float
    },
    "power_sample_size": int | None,

    # Meta-Analysis
    "meta_studies_df": pd.DataFrame,       # Study data for editing
    "meta_results": dict | None,           # Full analysis results
    "meta_measure_type": str,              # Selected measure type
    "meta_model": str,                     # Selected model (fixed/random/both)

    # Direct Standardization
    "direct_std_pop": str,               # Selected standard population name
    "direct_strata_df": pd.DataFrame,    # Editable strata table
    "direct_result": dict | None,        # Calculation result
    "direct_multiplier": str,            # Selected rate multiplier label

    # SMR/SIR Calculator
    "smr_mode": str,                     # "Simple (totals only)" or "Stratified ..."
    "smr_observed": int,                 # Simple mode: observed events
    "smr_expected": float,               # Simple mode: expected events
    "smr_result": dict | None,           # Simple mode result
    "smr_strata_df": pd.DataFrame,       # Stratified mode: editable strata table
    "smr_strat_result": dict | None,     # Stratified mode result

    # Data Analysis
    "data_df": pd.DataFrame | None,        # Uploaded dataset
    "data_source_name": str,               # Filename or "Pasted data"
    "data_col_summary": list[dict],        # Column summary from summarize_columns
    "data_outcome_col": str | None,        # Selected outcome column
    "data_exposure_col": str | None,       # Selected exposure column
    "data_confounder_cols": list[str],     # Selected confounder columns
    "data_outcome_positive": object,       # Positive outcome value
    "data_exposure_positive": object,      # Positive exposure value
    "data_weight_col": str | None,         # Survey weight column (optional)

    # Regression Analysis (Data Analysis Tab 5)
    "data_reg_model_type": str,            # Selected model: "Logistic (OR)" | "Linear (β)" | "Poisson (IRR)"
    "data_reg_result": dict | None,        # Last regression result

    # Propensity Score Analysis (Data Analysis Tab 6)
    "data_ps_estimand": str,               # "ATE" or "ATT"
    "data_ps_outcome_type": str,           # "Binary" or "Continuous"
    "data_ps_stabilized": bool,            # Stabilized weights toggle
    "data_ps_trim": float,                 # Trimming quantile (0-0.05)
    "data_ps_n_boot": int,                 # Bootstrap iterations
    "data_ps_result": dict | None,         # Full PS analysis result

    # Mediation Analysis (Data Analysis Tab 7)
    "data_mediator_cols": list[str],       # Selected mediator columns
    "data_med_result": dict | None,        # Full mediation analysis result

    # Cross-Tabulation Results (stored for PDF report)
    "data_crosstab_results": dict | None,  # {or, rr, rd, chi} from cross-tab tab
    "data_mh_result": dict | None,         # Mantel-Haenszel adjusted results

    # Export & Report (Page 8)
    "export_methods_text": str,            # Generated manuscript Methods section
}
```

## Scripts

### scripts/generate_demo_data.py

Generates a synthetic NHANES-style epidemiological dataset (250 rows, 9 columns:
age, sex, race, education, smoking, bmi, physical_activity, hypertension, survey_weight).
Output: `data/demo_epi.csv`. Uses logistic probability model for realistic exposure-outcome
relationships. All data is synthetic — no real NHANES records.

### scripts/diagnose_extraction.py

CLI tool for debugging Paper Analyzer extraction patterns.

**Usage:**
```bash
python scripts/diagnose_extraction.py              # Default: test_papers/
python scripts/diagnose_extraction.py ./my_papers  # Custom folder
```

**Outputs:**
- `extracted_text/{filename}.txt` - Normalized text with page markers
- `extraction_report.csv` - Per-pattern match counts for all PDFs
- `extraction_summary.txt` - Totals and pattern legend

**Pattern Tracking:**
- Tracks individual pattern matches (55 patterns total)
- Effect Measures: 20 patterns (OR: 6, HR: 3, RR: 3, PR: 3, IRR: 3, Beta: 2) + adjusted/crude detection
- CI: 8 patterns, P-value: 2 patterns, Sample: 4 patterns
- Beta Coefficients: 6 patterns, Mean Differences: 4 patterns, SD/SE: 6 patterns
- Weighted Statistics: 5 patterns (prevalence, mean, IPW, PS-weighted, weighted OR/HR/RR)
- Note: Per-pattern counts may exceed totals due to deduplication

## Cross-Page Integration

### DAG → Data Analysis (Feature A)
- Data Analysis Tab 2 checks for `dag_engine`, `dag_exposure`, `dag_outcome` in session state
- Calls `suggest_adjustment_set()` → `match_columns_to_dag_nodes()` to match DAG confounders to dataset columns
- Shows info banner with matched variables and "Apply DAG suggestions" button
- Button-triggered (not auto-applied) — user keeps full control of the multiselect

### Paper Analyzer → DAG Builder (Feature B)
- DAG Builder checks for `paper_results` in session state after confounder detection
- Filters to adjusted effect measures with `adjusted_for` fields
- Normalizes and compares paper adjustment set vs DAG adjustment set
- Shows overlap, DAG-only (warning), and paper-only (info) in an expander

## Testing

623 tests across 21 files. No Streamlit or network dependencies in any test.

### Coverage Map

| Test File | Source Module | Tests | Focus |
|-----------|--------------|-------|-------|
| `test_stats_calculator.py` | `stats_calculator.py` | 29 | OR, RR, RD, Chi-square, CI, NNT |
| `test_mh_calculator.py` | `stats_calculator.py` | 18 | Mantel-Haenszel adjusted OR/RR |
| `test_power_calculator.py` | `power_calculator.py` | 27 | Sample size, power, curves, OR sizing |
| `test_dag_engine.py` | `dag_engine.py` | 32 | DAG class API, serialization, queries |
| `test_confounder_detector.py` | `confounder_detector.py` | 21 | Graph analysis (confounders, paths, adjustment) |
| `test_confounder_matching.py` | `confounder_detector.py` | 15 | Name normalization, column matching |
| `test_paper_parser.py` | `paper_parser.py` | 55 | Regex extraction, PDF construction |
| `test_data_analyzer.py` | `data_analyzer.py` | 60 | Data upload, descriptive stats (enhanced), cross-tab |
| `test_meta_analysis.py` | `meta_analysis.py` | 36 | Fixed/random effects, heterogeneity |
| `test_regression.py` | `regression.py` | 22 | Logistic, linear, Poisson models |
| `test_propensity_score.py` | `propensity_score.py` | 50 | IPTW, balance, treatment effects |
| `test_mediation.py` | `mediation.py` | 42 | Baron-Kenny, indirect/direct effects |
| `test_e_value_integration.py` | `e_value.py` | 12 | E-value sensitivity analysis |
| `test_direct_standardization.py` | `direct_standardization.py` | 28 | Age-adjusted rates |
| `test_smr_calculator.py` | `smr_calculator.py` | 22 | Standardized mortality ratios |
| `test_report_generator.py` | `report_generator.py` | 28 | Methods section + PDF generation |
| `test_llm_extractor.py` | `llm_extractor.py` | 18 | Provider-agnostic LLM extraction |
| `test_llm_providers.py` | `llm_providers/` | 38 | Provider detection, parsing, Ollama/Gemini (mocked) |
| `test_methods_generator.py` | `methods_generator.py` | 39 | Methods text generation |
| `tests/utils/test_ui_helpers.py` | `utils/ui_helpers.py` | 14 | Styled banners, robustness badge, plot download |
| `test_smoke.py` | *(integration)* | 10 | Module imports + cross-module flows |

## Error Handling Strategy

1. **Input Validation**: All user inputs validated at UI layer before passing to core
2. **Graceful Degradation**: Show informative error messages, never crash
3. **Edge Cases**: Handle zeros, negative numbers, and invalid ranges explicitly
4. **Type Safety**: Use type hints and validate types at function boundaries
