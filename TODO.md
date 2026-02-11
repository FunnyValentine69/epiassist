# EpiAssist TODO

## Monetization Roadmap
- [ ] GitHub repo set to PRIVATE
- [ ] MIT license removed, proprietary copyright added
- [ ] Create public demo repo (Polyform Noncommercial license, basic extraction only)
- [ ] Add authentication (Streamlit Auth)
- [ ] Usage limits (free tier: 3 papers/month)
- [ ] Stripe integration
- [ ] Gumroad product page
- [ ] Landing page with demo video
- [ ] Beta testers (3-5 MPH students)
- [ ] Launch: r/epidemiology, r/publichealth, ProductHunt

## Completed

- [x] **Sprint 1: Project Skeleton**
  - Streamlit app structure, multipage navigation, theming

- [x] **Sprint 2: DAG Builder + Confounder Detection**
  - DAGEngine with NetworkX, Graphviz visualization
  - Backdoor path enumeration, adjustment set suggestions

- [x] **Sprint 3: Statistics Calculator**
  - 2x2 table analysis (OR, RR, RD, Chi-square)
  - Confidence intervals, plain English interpretations

- [x] **Sprint 4: Hypothesis Testing**
  - Research question formulation, null/alternative hypothesis
  - Study design recommendations, bias checklists

- [x] **Sprint 5: Paper Analyzer**
  - PDF text extraction with PyMuPDF
  - Regex patterns for OR, CI, p-values, sample sizes
  - Page-aware extraction with context snippets

- [x] **Sprint 6: Power Analysis + E-Value**
  - Sample size calculator, power curves
  - E-value sensitivity analysis for unmeasured confounding

- [x] **Paper Analyzer Enhancements**
  - Add HR, RR, PR, IRR extraction (5x improvement over OR-only)
  - Fix CI extraction: comma/bracket patterns (150% improvement)
  - Add β coefficient extraction
  - Add mean difference (MD) extraction
  - Add SD/SE extraction
  - Diagnostic scripts for pattern testing

- [x] **LLM-Enhanced Extraction (LangExtract + Ollama)**
  - Optional second-pass LLM extraction via local Ollama (llama3.1:8b)
  - LangExtract with few-shot examples for all 8 stat categories
  - Float-equal deduplication to merge regex + LLM results
  - Source column ("regex"/"llm") in all display tabs and CSV export
  - Toggle auto-disabled when Ollama not running

- [x] **Meta-Analysis + Forest Plot**
  - Inverse-variance fixed-effect and DerSimonian-Laird random-effects models
  - Forest plot (Plotly) with study weights and diamond pooled estimate
  - Funnel plot for publication bias assessment
  - Heterogeneity statistics (Q, I², τ²) with plain English interpretations
  - Import from Paper Analyzer (effect measures with complete CIs)
  - Supports ratio measures (OR, RR, HR, PR, IRR) on log scale and difference measures (MD, RD, beta)
  - 36 tests covering validation, pooling, heterogeneity, and full pipeline

- [x] **Phase 2: Data Analysis Module**
  - Data upload (CSV, Excel, paste) with auto-delimiter detection
  - Variable role assignment (outcome, exposure, confounders)
  - Column summary with numeric/categorical type heuristic
  - Descriptive statistics with grouped comparisons and Plotly histograms
  - Auto-generated 2x2 cross-tabulation via `build_contingency_table`
  - Reuses `stats_calculator` for OR, RR, RD, Chi-square — zero duplication
  - 30 tests across 6 classes, 93% coverage
  - openpyxl dependency added for Excel support

- [x] **Professional Repo Polish**
  - GitHub description and topics set via `gh` CLI
  - README badges (Python, Streamlit, License)
  - Issue templates (bug report, feature request)
  - Pull request template
  - Copyright year updated to 2025-2026

## Known Limitations

- **Paper Analyzer**
  - Table data extraction not implemented
  - Figure/chart data not extracted
  - LLM extraction requires local Ollama install (~10-15s/page)

- **Data Integration**
  - No NHANES data loader yet
  - No PubMed API integration

## Roadmap

### Medium-term
- [ ] NHANES data loader
- [ ] PubMed API integration for paper search
- [x] Meta-analysis calculator
- [x] Forest plot generation

### Long-term
- [ ] Manuscript generator (methods section templates)
- [ ] Propensity score calculator
- [ ] Mediation analysis module
- [ ] Export analysis sessions to PDF report

## Phase 2: Data Analysis Module (Complete)
- [x] Data upload interface (CSV, Excel, paste)
- [x] Variable selector (outcome, exposure, confounders)
- [x] Data preview and validation
- [x] Descriptive statistics (numeric + categorical, grouped by exposure)
- [x] Auto-generated 2x2 cross-tabulation with OR/RR/RD/Chi-square
- [x] 30 tests, 93% coverage on core/data_analyzer.py

## Phase 2.1: Basic Calculators (In Progress)
- [x] Mantel-Haenszel adjusted OR/RR
- [x] E-value calculator (unmeasured confounding) — auto-computed from crude/adjusted OR in Data Analysis Tab 4
- [ ] SMR/SIR calculator (standardized ratios)
- [ ] Direct/indirect standardization

## Phase 2.2: Regression Analysis (Planned)
- [ ] Logistic regression → adjusted ORs
- [ ] Linear regression → adjusted βs
- [ ] Poisson regression → adjusted IRRs

## Phase 2.3: Survey-Weighted Analysis (Planned)
- [ ] Weighted means/proportions
- [ ] Weighted regression with user-provided weight column

## Phase 3: Integration (Future)
- [ ] DAG → auto-suggest confounders for regression
- [ ] Paper extraction → compare to DAG recommendations
