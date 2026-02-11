# EpiAssist

**Epidemiological Research Assistant** - A Streamlit-based tool for causal inference, statistical analysis, and literature review in epidemiological research.

## Tech Stack

- Python 3.11+
- Streamlit (UI framework)
- NetworkX (graph operations)
- Graphviz (DAG visualization)
- SciPy (statistical functions)
- statsmodels (regression, statistical models)
- PyMuPDF (PDF parsing)
- Plotly (interactive charts)
- Pandas (data manipulation)
- LangExtract (optional — LLM-based stat extraction via Ollama)
- requests (HTTP client for Ollama health checks)

## Code Style

- **Formatter**: Black (default settings)
- **Type hints**: Required for all function signatures
- **Docstrings**: Required for all functions (Google style)
- **Imports**: stdlib, third-party, local (separated by blank lines)

## Architecture

```
epiassist/
├── app.py                    # Main entry point, navigation, theming
├── pages/                    # Streamlit UI components
│   ├── 1_DAG_Builder.py
│   ├── 2_Stats_Calculator.py
│   ├── 3_Hypothesis_Testing.py
│   ├── 4_Paper_Analyzer.py
│   ├── 5_Power_Analysis.py
│   ├── 6_Meta_Analysis.py
│   └── 7_Data_Analysis.py
├── core/                     # Business logic modules
│   ├── dag_engine.py
│   ├── stats_calculator.py   # OR, RR, RD, Chi-square, Mantel-Haenszel
│   ├── direct_standardization.py  # Direct standardization (Fay-Feuer CI)
│   ├── confounder_detector.py
│   ├── paper_parser.py
│   ├── llm_extractor.py      # Optional LLM extraction (LangExtract + Ollama)
│   ├── power_calculator.py
│   ├── e_value.py            # E-value sensitivity analysis (used by pages 5 & 7)
│   ├── meta_analysis.py      # Meta-analysis pooling (fixed/random effects)
│   ├── smr_calculator.py     # SMR/SIR calculator (exact Poisson CI)
│   ├── direct_standardization.py  # Direct standardization (age-adjusted rates, Fay-Feuer CI)
│   ├── regression.py          # Logistic, linear, Poisson regression (GLM)
│   └── data_analyzer.py      # Data upload, summary, contingency tables
├── utils/
│   ├── interpretations.py
│   └── constants.py
└── tests/                    # pytest test files
```

## Key Conventions

### Statistical Function Returns

All statistical functions must return a dict with this structure:

```python
{
    "value": float,           # Point estimate
    "ci_lower": float,        # Lower bound of 95% CI
    "ci_upper": float,        # Upper bound of 95% CI
    "interpretation": str     # Plain English explanation
}
```

### DAG Storage

- DAG nodes and edges stored as NetworkX DiGraph
- Access via `st.session_state.dag`
- Node attributes: `{"label": str, "type": "exposure"|"outcome"|"confounder"|"mediator"}`

### Session State

- Use `st.session_state` for all cross-page persistence
- Prefix keys by feature: `dag_`, `stats_`, `hypo_`, `paper_`, `power_`, `meta_`, `data_`, `smr_`, `direct_`, `data_reg_`

### Natural Language Interpretations

All statistical outputs must include plain English interpretations suitable for non-statisticians.

## Testing

- Framework: pytest
- Test files mirror source structure: `tests/core/test_stats_calculator.py`
- Run tests: `pytest tests/`
- Coverage target: 80%+ for core/ modules

## Common Commands

```bash
# Run the app
streamlit run app.py

# Run tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=core --cov=utils

# Format code
black .

# Type check
mypy core/ utils/
```

## Development Notes

- Always validate user inputs before statistical calculations
- Handle edge cases (zero counts, invalid ranges) gracefully with informative errors
- Prefer explicit over implicit - no magic numbers without constants
