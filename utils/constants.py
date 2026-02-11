"""Constants and reference values for epidemiological calculations."""

# Statistical defaults
ALPHA_DEFAULT: float = 0.05
POWER_DEFAULT: float = 0.80
CI_LEVEL_DEFAULT: float = 0.95
Z_SCORE_95: float = 1.96

# Effect size thresholds (Cohen's conventions)
EFFECT_SIZE_THRESHOLDS: dict[str, float] = {
    "small": 0.2,
    "medium": 0.5,
    "large": 0.8,
}

# Node colors for DAG visualization
NODE_COLORS: dict[str, str] = {
    "exposure": "#FF6B6B",  # Coral red
    "outcome": "#4ECDC4",   # Teal
    "confounder": "#FFE66D",  # Yellow
    "mediator": "#A78BFA",  # Purple
}

# Demo data for Hearing Loss and Unemployment study
DEMO_2X2_TABLE: dict[str, int] = {
    "a": 80,   # Exposed (HL), Outcome+ (Unemployed)
    "b": 150,  # Exposed (HL), Outcome- (Employed)
    "c": 70,   # Unexposed, Outcome+ (Unemployed)
    "d": 400,  # Unexposed, Outcome- (Employed)
}

# Demo DAG configuration
DEMO_DAG_NODES: list[dict[str, str]] = [
    {"name": "Hearing Loss", "type": "exposure"},
    {"name": "Unemployment", "type": "outcome"},
    {"name": "Age", "type": "confounder"},
    {"name": "Education", "type": "confounder"},
    {"name": "Race", "type": "confounder"},
    {"name": "Depression", "type": "confounder"},
    {"name": "Other Disabilities", "type": "confounder"},
]

# Meta-analysis constants
I_SQUARED_THRESHOLDS: dict[str, float] = {
    "low": 25.0,
    "moderate": 50.0,
    "high": 75.0,
}

RATIO_MEASURES: set[str] = {"OR", "RR", "HR", "PR", "IRR"}

DIFFERENCE_MEASURES: set[str] = {"MD", "RD", "beta"}

META_MEASURE_LABELS: dict[str, str] = {
    "OR": "Odds Ratio",
    "RR": "Risk Ratio",
    "HR": "Hazard Ratio",
    "PR": "Prevalence Ratio",
    "IRR": "Incidence Rate Ratio",
    "MD": "Mean Difference",
    "RD": "Risk Difference",
    "beta": "Beta Coefficient",
}

DEMO_DAG_EDGES: list[tuple[str, str]] = [
    ("Age", "Hearing Loss"),
    ("Age", "Unemployment"),
    ("Education", "Hearing Loss"),
    ("Education", "Unemployment"),
    ("Race", "Hearing Loss"),
    ("Race", "Unemployment"),
    ("Depression", "Hearing Loss"),
    ("Depression", "Unemployment"),
    ("Other Disabilities", "Hearing Loss"),
    ("Other Disabilities", "Unemployment"),
    ("Hearing Loss", "Unemployment"),
]

# Standard populations for direct standardization
# Weights represent the standard population count in each age group.
STANDARD_POPULATIONS: dict[str, list[dict]] = {
    "US 2000": [
        {"stratum_name": "0-4", "weight": 18987},
        {"stratum_name": "5-14", "weight": 39977},
        {"stratum_name": "15-24", "weight": 38077},
        {"stratum_name": "25-34", "weight": 37233},
        {"stratum_name": "35-44", "weight": 44659},
        {"stratum_name": "45-54", "weight": 37030},
        {"stratum_name": "55-64", "weight": 23961},
        {"stratum_name": "65-74", "weight": 18136},
        {"stratum_name": "75-84", "weight": 12315},
        {"stratum_name": "85+", "weight": 4259},
    ],
    "WHO World": [
        {"stratum_name": "0-4", "weight": 8860},
        {"stratum_name": "5-14", "weight": 17020},
        {"stratum_name": "15-24", "weight": 17020},
        {"stratum_name": "25-34", "weight": 15020},
        {"stratum_name": "35-44", "weight": 12170},
        {"stratum_name": "45-54", "weight": 9610},
        {"stratum_name": "55-64", "weight": 7150},
        {"stratum_name": "65-74", "weight": 5410},
        {"stratum_name": "75-84", "weight": 3520},
        {"stratum_name": "85+", "weight": 1520},
    ],
    "Segi World": [
        {"stratum_name": "0-4", "weight": 12000},
        {"stratum_name": "5-14", "weight": 18000},
        {"stratum_name": "15-24", "weight": 17000},
        {"stratum_name": "25-34", "weight": 14000},
        {"stratum_name": "35-44", "weight": 12000},
        {"stratum_name": "45-54", "weight": 9000},
        {"stratum_name": "55-64", "weight": 6000},
        {"stratum_name": "65-74", "weight": 4000},
        {"stratum_name": "75-84", "weight": 2000},
        {"stratum_name": "85+", "weight": 1000},
    ],
}

# Rate multipliers for direct standardization output
RATE_MULTIPLIERS: dict[str, int] = {
    "per 1,000": 1_000,
    "per 10,000": 10_000,
    "per 100,000": 100_000,
}
