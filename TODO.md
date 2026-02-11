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
- [ ] Meta-analysis calculator
- [ ] Forest plot generation

### Long-term
- [ ] Manuscript generator (methods section templates)
- [ ] Propensity score calculator
- [ ] Mediation analysis module
- [ ] Export analysis sessions to PDF report

## Phase 2: Data Analysis Module (Planned)
- [ ] Data upload interface (CSV, Excel, paste)
- [ ] Variable selector (outcome, exposure, confounders, weights)
- [ ] Data preview and validation

## Phase 2.1: Basic Calculators (Planned)
- [ ] Mantel-Haenszel adjusted OR/RR
- [ ] E-value calculator (unmeasured confounding)
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
