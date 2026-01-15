# Contributing to EpiAssist

## Development Setup

### Prerequisites

- Python 3.11 or higher
- Graphviz (for DAG visualization)

### Installation

**macOS:**
```bash
brew install graphviz
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install graphviz
```

**Windows:**
Download from https://graphviz.org/download/ and add to PATH.

### Project Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/epiassist.git
cd epiassist

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=core --cov=utils

# Run specific test file
pytest tests/core/test_paper_parser.py
```

## Code Style

### Formatting

We use **Black** for code formatting with default settings:

```bash
black .
```

### Type Hints

Type hints are **required** for all function signatures:

```python
def calculate_odds_ratio(a: int, b: int, c: int, d: int) -> dict:
    ...
```

### Docstrings

Use **Google-style** docstrings:

```python
def find_odds_ratios(text: str, page: int = 1) -> list[dict]:
    """Find odds ratios mentioned in text.

    Args:
        text: Text to search.
        page: Page number where this text was found.

    Returns:
        List of dicts with 'value', 'ci_lower', 'ci_upper', 'context', 'page'.
    """
```

### Type Checking

```bash
mypy core/ utils/
```

## Adding New Extraction Patterns to Paper Analyzer

The Paper Analyzer extracts statistics from PDF text using regex patterns.
Patterns are defined in `core/paper_parser.py`.

### Pattern Structure

Patterns are organized in cascading lists, **most specific first**:

```python
patterns = [
    r"...",  # Most specific (e.g., OR with full CI label)
    r"...",  # Less specific (e.g., OR with CI abbreviation)
    r"...",  # Least specific (e.g., standalone OR)
]
```

### Adding a New Pattern

1. **Identify the format** in actual papers (collect examples)

2. **Write the regex pattern** with capture groups for values:
   ```python
   # Example: HR 1.45 (95% CI 1.12-1.89)
   r"(?:hr|hazard\s+ratio)[,:\s=]+(\d+(?:\.\d+)?)\s*\(.*?(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\)"
   ```

3. **Test the pattern in isolation**:
   ```python
   import re
   text = "HR 1.45 (95% CI 1.12-1.89)"
   match = re.search(pattern, text, re.IGNORECASE)
   print(match.groups())  # ('1.45', '1.12', '1.89')
   ```

4. **Add to the appropriate function** in `paper_parser.py`

5. **Validate with the diagnostic script**:
   ```bash
   python scripts/diagnose_extraction.py test_papers/
   ```

### Pattern Conventions

- Use `(?:...)` for non-capturing groups
- Use `\d+(?:\.\d+)?` for numbers (matches "2" or "2.5")
- Use `(?:-|to)` for range separators
- Use `[,:\s=]+` for flexible delimiters
- Always use `re.IGNORECASE` flag

### Testing Patterns

1. Add test PDFs to `test_papers/`
2. Run the diagnostic script
3. Check `extraction_report.csv` for pattern hit counts
4. Check `missed_patterns.txt` for uncaptured statistics

## Project Structure

```
epiassist/
├── app.py                 # Main entry point
├── pages/                 # Streamlit UI pages
├── core/                  # Business logic modules
├── utils/                 # Shared utilities
├── scripts/               # CLI tools
│   └── diagnose_extraction.py
├── tests/                 # pytest test files
└── test_papers/           # Test PDFs (gitignored)
```

See `ARCHITECTURE.md` for detailed module documentation.
