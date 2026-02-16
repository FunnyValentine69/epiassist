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
- [x] Manuscript generator (methods section templates) — Page 8 Export & Report
- [x] Propensity score calculator
- [x] Mediation analysis module
- [x] Export analysis sessions to PDF report — Page 8 Export & Report

## Phase 2: Data Analysis Module (Complete)
- [x] Data upload interface (CSV, Excel, paste)
- [x] Variable selector (outcome, exposure, confounders)
- [x] Data preview and validation
- [x] Descriptive statistics (numeric + categorical, grouped by exposure)
- [x] Auto-generated 2x2 cross-tabulation with OR/RR/RD/Chi-square
- [x] 30 tests, 93% coverage on core/data_analyzer.py

## Phase 2.4: Propensity Score Analysis (Complete)
- [x] Propensity score estimation via logistic regression (GLM Binomial)
- [x] IPTW weight calculation (ATE/ATT, stabilized, optional trimming)
- [x] Common support assessment (overlap % with warning)
- [x] Balance diagnostics with SMD threshold (0.1) and Love plot
- [x] Treatment effect estimation with bootstrap CIs (binary OR, continuous mean diff)
- [x] Survey weight integration (IPTW * survey weight)
- [x] E-value integration for residual unmeasured confounding
- [x] Data Analysis Tab 6 with full UI (settings, histograms, Love plot, metrics)
- [x] 50 tests across 9 classes covering all functions and interpretations

## Phase 2.5: Mediation Analysis (Complete)
- [x] Baron-Kenny 3-step regression (total, a-path, direct models) via `_fit_glm`
- [x] Indirect/direct/total effect decomposition (product method for continuous, difference method for binary)
- [x] Sobel test for continuous outcomes, bootstrap percentile CIs for all
- [x] Proportion mediated with sign-check guard
- [x] Survey weight integration (passthrough to `_fit_glm`)
- [x] DAG-based mediator auto-suggestion in Data Analysis Tab 2
- [x] Data Analysis Tab 7 with full UI (settings, effect decomposition, path coefficients, interpretation)
- [x] 42 tests across 6 classes covering data prep, model fitting, effects, bootstrap, full pipeline, interpretation

## Phase 2.1: Basic Calculators (In Progress)
- [x] Mantel-Haenszel adjusted OR/RR
- [x] E-value calculator (unmeasured confounding) — auto-computed from crude/adjusted OR in Data Analysis Tab 4
- [x] SMR/SIR calculator (standardized ratios)
- [x] Direct/indirect standardization

## Phase 2.2: Regression Analysis (Complete)
- [x] Logistic regression → adjusted ORs (GLM Binomial)
- [x] Linear regression → adjusted βs (GLM Gaussian)
- [x] Poisson regression → adjusted IRRs (GLM Poisson)
- [x] Data Analysis Tab 5 with model selection, coefficient tables, model fit stats
- [x] 22 tests across 6 classes covering data prep, all 3 models, and interpretations

## Phase 2.3: Survey-Weighted Analysis (Complete)
- [x] Weighted means/proportions (weighted_stats_numeric, weighted_stats_categorical, grouped_weighted)
- [x] Weighted regression with user-provided weight column (freq_weights via GLM)
- [x] Weight column role assignment in Data Analysis Tab 2 with validation (> 0)
- [x] Weighted descriptive stats in Tab 3 with info banner and effective N (Kish's formula)
- [x] Survey-weighted interpretation prefix in regression output
- [x] 19 new tests (9 data_analyzer + 10 regression weighted tests)

## Phase 3: DAG + Paper Integration (Complete)
- [x] DAG → auto-suggest confounders in Data Analysis Tab 2 (via `match_columns_to_dag_nodes`)
- [x] Paper extraction → compare adjustment sets on DAG Builder page
- [x] Name normalization functions (`normalize_variable_name`, `match_columns_to_dag_nodes`)
- [x] 15 tests for matching functions
