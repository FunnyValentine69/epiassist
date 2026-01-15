# EpiAssist Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            STREAMLIT UI                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │   DAG    │ │  Stats   │ │Hypothesis│ │  Paper   │ │  Power   │      │
│  │ Builder  │ │Calculator│ │ Testing  │ │ Analyzer │ │ Analysis │      │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘      │
│       │            │            │            │            │             │
└───────┼────────────┼────────────┼────────────┼────────────┼─────────────┘
        │            │            │            │            │
        ▼            ▼            ▼            ▼            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CORE MODULES                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │   dag    │ │  stats   │ │confounder│ │  paper   │ │  power   │      │
│  │ engine   │ │calculator│ │ detector │ │  parser  │ │calculator│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                          ┌──────────┐                   │
│                                          │ e_value  │                   │
│                                          └──────────┘                   │
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
- 2x2 contingency table input
- Odds Ratio, Risk Ratio, Risk Difference calculations
- Confidence interval display
- Chi-square test results
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
```

#### confounder_detector.py
```python
def find_confounders(dag: nx.DiGraph, exposure: str, outcome: str) -> list[str]
def find_backdoor_paths(dag: nx.DiGraph, exposure: str, outcome: str) -> list[list[str]]
def suggest_adjustment_set(dag: nx.DiGraph, exposure: str, outcome: str) -> list[str]
```

#### paper_parser.py
```python
def extract_text_from_pdf(file_bytes: bytes) -> list[tuple[int, str]]
def find_effect_measures(text: str, page: int = 1) -> list[dict]
    # Returns dicts with: type (OR|HR|RR|PR|IRR|β), value, ci_lower, ci_upper, context, page
def find_confidence_intervals(text: str, page: int = 1) -> list[dict]
def find_p_values(text: str, page: int = 1) -> list[dict]
def find_sample_sizes(text: str, page: int = 1) -> list[dict]
```

#### power_calculator.py
```python
def calculate_sample_size(effect_size: float, alpha: float, power: float) -> int
def calculate_power(n: int, effect_size: float, alpha: float) -> float
def generate_power_curve(effect_size: float, alpha: float, n_range: tuple[int, int]) -> pd.DataFrame
```

#### e_value.py
```python
def calculate_e_value(point_estimate: float, ci_bound: float = None) -> dict
def interpret_e_value(e_value: float) -> str
```

### utils/ (Shared Utilities)

#### interpretations.py
```python
def interpret_odds_ratio(or_value: float, ci_lower: float, ci_upper: float) -> str
def interpret_risk_ratio(rr_value: float, ci_lower: float, ci_upper: float) -> str
def interpret_p_value(p: float, alpha: float = 0.05) -> str
def interpret_power(power: float) -> str
def interpret_e_value(e_value: float) -> str
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
│ PDF Upload  │───▶│  PyMuPDF    │───▶│   Regex     │───▶│  Extracted  │
│             │    │  Extract    │    │  Patterns   │    │   Stats     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                               │
                         ┌─────────────────────────────────────┤
                         ▼                   ▼                 ▼
                   ┌──────────┐       ┌──────────┐      ┌──────────┐
                   │ ORs/RRs  │       │   CIs    │      │ P-values │
                   └──────────┘       └──────────┘      └──────────┘
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
    "power_sample_size": int | None
}
```

## Scripts

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
- Tracks individual pattern matches (32 patterns total)
- Effect Measures: 20 patterns (OR: 6, HR: 3, RR: 3, PR: 3, IRR: 3, Beta: 2)
- CI: 6 patterns, P-value: 2 patterns, Sample: 4 patterns
- Note: Per-pattern counts may exceed totals due to deduplication

## Error Handling Strategy

1. **Input Validation**: All user inputs validated at UI layer before passing to core
2. **Graceful Degradation**: Show informative error messages, never crash
3. **Edge Cases**: Handle zeros, negative numbers, and invalid ranges explicitly
4. **Type Safety**: Use type hints and validate types at function boundaries
