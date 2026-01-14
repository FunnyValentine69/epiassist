# EpiAssist

An epidemiological research assistant for causal inference, statistical analysis, and literature review.

## Features

- **DAG Builder**: Create and visualize Directed Acyclic Graphs for causal inference. Automatically detect confounders and suggest adjustment sets using backdoor criterion.

- **Statistics Calculator**: Calculate Odds Ratios, Risk Ratios, Risk Differences, and Chi-square tests from 2x2 contingency tables with confidence intervals and plain English interpretations.

- **Hypothesis Framework**: Structure your research questions into formal null and alternative hypotheses. Get study design recommendations and bias checklists.

- **Paper Analyzer**: Upload epidemiological papers (PDF) and automatically extract reported statistics including odds ratios, confidence intervals, p-values, and sample sizes.

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
- **Plotly** - Interactive visualizations
- **Pandas** - Data manipulation

## Development

```bash
# Run tests
pytest tests/

# Format code
black .

# Type check
mypy core/ utils/
```

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
