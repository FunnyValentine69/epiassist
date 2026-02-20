"""Shared extraction prompts used by all LLM providers."""

EXTRACTION_SYSTEM_PROMPT = (
    "You are an epidemiological statistics extraction engine. "
    "Given text from a research paper, extract ALL statistical results "
    "and return them as a JSON object with these exact keys:\n"
    '  "effect_measures": [{{"type": "OR|HR|RR|PR|IRR", "value": float, '
    '"ci_lower": float|null, "ci_upper": float|null, "context": str}}]\n'
    '  "confidence_intervals": [{{"level": int, "lower": float, "upper": float, "context": str}}]\n'
    '  "p_values": [{{"value": float, "operator": "=|<|>", "context": str}}]\n'
    '  "sample_sizes": [{{"value": int}}]\n'
    '  "beta_coefficients": [{{"value": float, "ci_lower": float|null, '
    '"ci_upper": float|null, "se": float|null, "context": str}}]\n'
    '  "mean_differences": [{{"value": float, "ci_lower": float|null, '
    '"ci_upper": float|null, "context": str}}]\n'
    '  "standard_deviations": [{{"value": float, "mean": float|null, '
    '"type": "SD|SE", "context": str}}]\n'
    '  "weighted_statistics": [{{"stat_type": str, "value": float, '
    '"weight_method": str|null, "context": str}}]\n'
    "Return ONLY valid JSON. Every key must be present (use empty list [] if none found). "
    "Extract numeric values only — do not invent data."
)

FEW_SHOT_USER = (
    "The adjusted odds ratio was 2.45 (95% CI: 1.12-5.34, p=0.024). "
    "HR = 1.78 (1.23-2.56) after adjusting for age and sex. "
    "Beta coefficient: 0.34 (95% CI: 0.12, 0.56). "
    "Mean difference was 3.2 (SD 1.5), n=450 participants."
)

FEW_SHOT_ASSISTANT = """{
  "effect_measures": [
    {"type": "OR", "value": 2.45, "ci_lower": 1.12, "ci_upper": 5.34, "context": "adjusted odds ratio was 2.45 (95% CI: 1.12-5.34"},
    {"type": "HR", "value": 1.78, "ci_lower": 1.23, "ci_upper": 2.56, "context": "HR = 1.78 (1.23-2.56)"}
  ],
  "confidence_intervals": [],
  "p_values": [
    {"value": 0.024, "operator": "=", "context": "p=0.024"}
  ],
  "sample_sizes": [
    {"value": 450}
  ],
  "beta_coefficients": [
    {"value": 0.34, "ci_lower": 0.12, "ci_upper": 0.56, "se": null, "context": "Beta coefficient: 0.34 (95% CI: 0.12, 0.56)"}
  ],
  "mean_differences": [
    {"value": 3.2, "ci_lower": null, "ci_upper": null, "context": "Mean difference was 3.2"}
  ],
  "standard_deviations": [
    {"value": 1.5, "mean": 3.2, "type": "SD", "context": "SD 1.5"}
  ],
  "weighted_statistics": []
}"""
