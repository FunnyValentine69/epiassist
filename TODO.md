# EpiAssist Development TODO

## Sprint 1: Project Skeleton
**Goal**: Basic Streamlit app with navigation and structure

- [ ] Create `app.py` with page config and welcome content
- [ ] Set up multipage structure in `pages/`
- [ ] Create placeholder pages (DAG, Stats, Hypothesis, Paper, Power)
- [ ] Initialize `core/` module structure with `__init__.py`
- [ ] Initialize `utils/` module structure
- [ ] Set up `tests/` directory structure
- [ ] Configure basic theming and sidebar navigation
- [ ] Verify app runs with `streamlit run app.py`

---

## Sprint 2: DAG Builder + Confounder Detection
**Goal**: Interactive DAG creation with causal analysis

- [ ] Implement `core/dag_engine.py`
  - [ ] DAGEngine class with NetworkX DiGraph
  - [ ] add_node(), remove_node() methods
  - [ ] add_edge(), remove_edge() methods
  - [ ] get_all_paths() for path enumeration
  - [ ] render_graphviz() for visualization
  - [ ] to_dict() / from_dict() for serialization
- [ ] Implement `core/confounder_detector.py`
  - [ ] find_confounders() - identify common causes
  - [ ] find_backdoor_paths() - enumerate backdoor paths
  - [ ] suggest_adjustment_set() - minimal adjustment set
- [ ] Build `pages/1_DAG_Builder.py`
  - [ ] Node creation form (name, type dropdown)
  - [ ] Edge creation form (source, target dropdowns)
  - [ ] DAG visualization display
  - [ ] Exposure/outcome selector
  - [ ] Confounder detection button and results
  - [ ] Session state persistence
- [ ] Add `utils/constants.py` with NODE_COLORS
- [ ] Write tests for dag_engine.py
- [ ] Write tests for confounder_detector.py

---

## Sprint 3: Statistics Calculator + Interpretations
**Goal**: 2x2 table analysis with natural language output

- [ ] Implement `core/stats_calculator.py`
  - [ ] calculate_odds_ratio() with CI
  - [ ] calculate_risk_ratio() with CI
  - [ ] calculate_risk_difference() with CI
  - [ ] calculate_chi_square() with p-value
  - [ ] calculate_confidence_interval() helper
- [ ] Implement `utils/interpretations.py`
  - [ ] interpret_odds_ratio()
  - [ ] interpret_risk_ratio()
  - [ ] interpret_risk_difference()
  - [ ] interpret_p_value()
  - [ ] interpret_chi_square()
- [ ] Build `pages/2_Stats_Calculator.py`
  - [ ] 2x2 table input (4 number inputs)
  - [ ] Calculate button
  - [ ] Results display (OR, RR, RD, Chi-square)
  - [ ] Interpretation text boxes
  - [ ] Input validation (positive integers)
- [ ] Write tests for stats_calculator.py
- [ ] Write tests for interpretations.py

---

## Sprint 4: Hypothesis Testing Framework
**Goal**: Structured hypothesis formulation and study design guidance

- [ ] Build `pages/3_Hypothesis_Testing.py`
  - [ ] Research question text input
  - [ ] Auto-generate null hypothesis
  - [ ] Auto-generate alternative hypothesis
  - [ ] Study design selector (cohort, case-control, cross-sectional, RCT)
  - [ ] Design-specific guidance display
  - [ ] Bias checklist (selection, information, confounding)
  - [ ] PICO framework helper (Population, Intervention, Comparison, Outcome)
- [ ] Add study design templates to constants.py
- [ ] Add bias type descriptions to constants.py

---

## Sprint 5: Paper Analyzer
**Goal**: Extract statistics from uploaded PDFs using regex patterns

- [ ] Implement `core/paper_parser.py`
  - [ ] extract_text_from_pdf() using PyMuPDF
  - [ ] find_odds_ratios() - regex for OR patterns
  - [ ] find_confidence_intervals() - regex for CI patterns
  - [ ] find_p_values() - regex for p-value patterns
  - [ ] find_sample_sizes() - regex for n= patterns
- [ ] Build `pages/4_Paper_Analyzer.py`
  - [ ] PDF file uploader
  - [ ] Processing status indicator
  - [ ] Extracted statistics table
  - [ ] Raw text preview (collapsible)
  - [ ] Export findings button
- [ ] Create comprehensive regex patterns for epi stats
- [ ] Write tests for paper_parser.py with sample text

---

## Sprint 6: Power Analysis + E-Value
**Goal**: Sample size planning and sensitivity analysis

- [ ] Implement `core/power_calculator.py`
  - [ ] calculate_sample_size()
  - [ ] calculate_power()
  - [ ] generate_power_curve()
- [ ] Implement `core/e_value.py`
  - [ ] calculate_e_value()
  - [ ] interpret_e_value()
- [ ] Build `pages/5_Power_Analysis.py`
  - [ ] Effect size input
  - [ ] Alpha level selector
  - [ ] Desired power slider
  - [ ] Sample size result display
  - [ ] Power curve plot (Plotly)
  - [ ] E-value calculator section
  - [ ] E-value interpretation
- [ ] Add interpret_power() to interpretations.py
- [ ] Add interpret_e_value() to interpretations.py
- [ ] Write tests for power_calculator.py
- [ ] Write tests for e_value.py

---

## Sprint 7: Polish and Testing
**Goal**: Final cleanup, testing, and documentation

- [ ] Run full test suite, fix failures
- [ ] Add input validation across all pages
- [ ] Improve error messages and edge case handling
- [ ] Add tooltips/help text for complex concepts
- [ ] Test cross-page navigation and state persistence
- [ ] Verify all interpretations are accurate
- [ ] Add example data / demo mode
- [ ] Update README with final screenshots
- [ ] Code review and cleanup
- [ ] Final manual testing of all features

---

## Future Enhancements (Backlog)

- [ ] Export results to PDF report
- [ ] Save/load analysis sessions
- [ ] Meta-analysis calculator
- [ ] Forest plot generation
- [ ] Integration with PubMed API for paper search
- [ ] DAG import from DAGitty format
- [ ] Propensity score calculator
- [ ] Mediation analysis module
