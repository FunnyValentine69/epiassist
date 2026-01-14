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
