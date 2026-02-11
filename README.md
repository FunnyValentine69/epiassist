# EpiAssist

An epidemiological research assistant for causal inference, statistical analysis, and literature review.

## Features

- **DAG Builder**: Create and visualize Directed Acyclic Graphs for causal inference. Automatically detect confounders and suggest adjustment sets using backdoor criterion.

- **Statistics Calculator**: Calculate Odds Ratios, Risk Ratios, Risk Differences, and Chi-square tests from 2x2 contingency tables with confidence intervals and plain English interpretations.

- **Hypothesis Framework**: Structure your research questions into formal null and alternative hypotheses. Get study design recommendations and bias checklists.

- **Paper Analyzer**: Upload epidemiological papers (PDF) and automatically extract:
  - Effect measures (OR, HR, RR, PR, IRR) with **adjusted vs crude detection**
  - Adjustment variables (e.g., "adjusted for age, sex, education")
  - Beta coefficients (β, B, coefficient)
  - Confidence intervals (95% CI)
  - P-values
  - Mean differences (MD)
  - Standard deviations and standard errors (SD, SE)
  - Weighted statistics (IPW, survey-weighted, PS-weighted)
  - Sample sizes

  Includes page citations for each extracted statistic. Handles PDF text quirks (hyphenation, en-dash, Unicode encoding). Validated against published research.

- **Power Analysis**: Calculate required sample sizes, generate power curves, and perform E-value sensitivity analysis for unmeasured confounding.

## Installation

### Prerequisites

- Python 3.11 or higher
- Graphviz (for DAG visualization)

### macOS

```bash
# Install Graphviz
brew install graphviz

# Clone the repository
git clone https://github.com/yourusername/epiassist.git
cd epiassist

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Linux (Ubuntu/Debian)

```bash
# Install Graphviz
sudo apt-get install graphviz

# Clone and setup (same as above)
git clone https://github.com/yourusername/epiassist.git
cd epiassist
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Windows

```bash
# Install Graphviz from https://graphviz.org/download/
# Add Graphviz bin directory to PATH

# Clone and setup
git clone https://github.com/yourusername/epiassist.git
cd epiassist
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Screenshots

<!-- TODO: Add screenshots -->
*Screenshots coming soon*

## Demo Use Case: Hearing Loss and Employment Status

Investigate the relationship between hearing loss and employment status using NHANES data:

1. **Build a DAG**: Create nodes for Hearing Loss (exposure), Employment Status (outcome), and potential confounders like Age, Education, and Occupation Type.

2. **Detect Confounders**: Use the confounder detection to identify which variables create backdoor paths that need adjustment.

3. **Calculate Statistics**: Input your 2x2 table from NHANES analysis to calculate adjusted odds ratios.

4. **Power Analysis**: Determine if your sample size provides adequate power to detect the expected effect.

5. **E-Value**: Assess how robust your findings are to potential unmeasured confounding.

## Tech Stack

- **Streamlit** - Web application framework
- **NetworkX** - Graph operations for DAG analysis
- **Graphviz** - DAG visualization
- **SciPy / statsmodels** - Statistical calculations
- **PyMuPDF** - PDF text extraction
- **LangExtract** - Optional LLM-based stat extraction
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

## Optional: AI-Enhanced Extraction (Ollama)

The Paper Analyzer can optionally use a local LLM via [Ollama](https://ollama.com) to catch statistics that regex patterns miss. This is completely free and runs locally.

### Setup

```bash
# Install Ollama (macOS)
brew install ollama

# Pull the model (~4.7GB download)
ollama pull llama3.1:8b

# Start the server (keep running in a separate terminal)
ollama serve
```

### Usage

1. Start Ollama (`ollama serve`)
2. Open Paper Analyzer in EpiAssist
3. Check the "Enhance with AI (Ollama)" checkbox
4. Upload a PDF and extract — the LLM runs a second pass after regex
5. New results show "llm" in the Source column

When Ollama isn't running, the checkbox is automatically disabled.

## Development

```bash
# Run tests
pytest tests/

# Format code
black .

# Type check
mypy core/ utils/
```

---

Copyright © 2025 Rogue Semicolon. All rights reserved.
