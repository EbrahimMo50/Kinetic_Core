import numpy as np
import pandas as pd

def make_sample_df(n_rows: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic customer-churn-style dataset that triggers
    every preprocessing step:
      - Missing values in both numeric and categorical columns
      - Categorical columns with varying cardinality
      - Numeric columns at different scales
      - Imbalanced binary target (~10% positive class)
    """
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame({
        # Numeric features at different scales (forces the need for scaling)
        "age":              rng.integers(18, 80, size=n_rows),
        "annual_income":    rng.normal(20_000, 55_000, size=n_rows).round(2),
        "account_balance":  rng.exponential(3_000, size=n_rows).round(2),
        "tenure_months":    rng.integers(0, 120, size=n_rows),
        "num_products":     rng.integers(1, 6, size=n_rows),

        # Categorical features (force one-hot encoding)
        "country":      rng.choice(["US", "UK", "DE", "FR", "JP"], size=n_rows,
                                   p=[0.4, 0.2, 0.15, 0.15, 0.1]),
        "plan_type":    rng.choice(["basic", "standard", "premium"], size=n_rows,
                                   p=[0.5, 0.35, 0.15]),
        "is_active":    rng.choice([True, False], size=n_rows, p=[0.7, 0.3]),

        # Imbalanced target (~10% churn — triggers SMOTE)
        "churned":      rng.choice([0, 1], size=n_rows, p=[0.9, 0.1]),
    })

    # Sprinkle in missing values (~5% per affected column)
    for col, missing_rate in [
        ("annual_income", 0.05),
        ("account_balance", 0.08),
        ("country", 0.03),
        ("plan_type", 0.04),
    ]:
        mask = rng.random(n_rows) < missing_rate
        df.loc[mask, col] = np.nan

    return df