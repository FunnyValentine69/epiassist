# EpiAssist TODO

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

## Known Limitations

- **Paper Analyzer**
  - Table data extraction not implemented
  - Figure/chart data not extracted
  - Some papers use non-standard formats

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
