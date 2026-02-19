"""Generate a synthetic NHANES-style epidemiological demo dataset.

Creates ~250 rows with demographics, exposure, outcome, confounders,
and survey weights. All data is synthetic — no real NHANES records.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 250

# Demographics
age = np.random.normal(50, 15, N).clip(18, 85).astype(int)
sex = np.random.choice(["Male", "Female"], N, p=[0.48, 0.52])
race = np.random.choice(
    ["White", "Black", "Hispanic", "Asian", "Other"],
    N,
    p=[0.60, 0.13, 0.18, 0.06, 0.03],
)
education = np.random.choice(
    ["< High School", "High School", "Some College", "College+"],
    N,
    p=[0.12, 0.27, 0.29, 0.32],
)

# Exposure: smoking status (binary)
# Higher probability for older, lower education
smoke_prob = 0.15 + 0.003 * (age - 50)
smoke_prob += np.where(education == "< High School", 0.10, 0.0)
smoke_prob += np.where(education == "High School", 0.05, 0.0)
smoke_prob = np.clip(smoke_prob, 0.05, 0.60)
smoking = np.random.binomial(1, smoke_prob).astype(str)
smoking = np.where(smoking == "1", "Current", "Never")

# Confounder: BMI
bmi = np.random.normal(28, 5, N).clip(16, 50).round(1)

# Outcome: hypertension (binary)
# Influenced by age, smoking, BMI
hyp_prob = -3.0 + 0.04 * age + 0.5 * (smoking == "Current") + 0.05 * bmi
hyp_prob = 1 / (1 + np.exp(-hyp_prob))  # logistic
hypertension = np.random.binomial(1, hyp_prob).astype(str)
hypertension = np.where(hypertension == "1", "Yes", "No")

# Mediator: physical activity (could mediate smoking → hypertension)
activity = np.random.choice(
    ["Sedentary", "Moderate", "Active"],
    N,
    p=[0.30, 0.45, 0.25],
)
# Smokers slightly more sedentary
activity = np.where(
    (smoking == "Current") & (np.random.random(N) < 0.3),
    "Sedentary",
    activity,
)

# Survey weights (simulate complex survey design)
weights = np.random.lognormal(0, 0.5, N).round(2)
weights = (weights / weights.mean() * 1.0).round(4)  # normalize around 1

df = pd.DataFrame({
    "age": age,
    "sex": sex,
    "race": race,
    "education": education,
    "smoking": smoking,
    "bmi": bmi,
    "physical_activity": activity,
    "hypertension": hypertension,
    "survey_weight": weights,
})

df.to_csv("data/demo_epi.csv", index=False)
print(f"Generated {len(df)} rows, {len(df.columns)} columns")
print(f"Smoking: {(df['smoking'] == 'Current').mean():.1%}")
print(f"Hypertension: {(df['hypertension'] == 'Yes').mean():.1%}")
print(df.head())
