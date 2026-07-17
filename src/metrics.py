"""
metrics.py
----------
Computes the "gap_score": how far above or below expected maternal
mortality a country sits, given its level of gender inequality (GII or
GDI, whichever the user selects in the dashboard).

Methodology (same approach as the healthcare-expenditure dashboard):
- Fits a simple linear regression: log(maternal_mortality) ~ indicator.
  log(maternal mortality) is used because its distribution is heavily
  skewed (countries range from <5 to >1500 deaths per 100k live births) —
  without the log transform, a handful of extreme-value countries would
  dominate the fit.
- gap_score = residual of that regression.
    - Score > 0: the country has MORE maternal mortality than its gender
      inequality level would predict (worse than expected).
    - Score < 0: LESS than expected (better than expected).

IMPORTANT — direction of the relationship depends on the indicator:
- GII (0=parity, 1=high inequality): a POSITIVE relationship with
  maternal mortality is expected (more inequality -> more mortality).
- GDI (~1=parity, >1 women exceed men's HDI, <1 the reverse): its
  relationship with maternal mortality isn't necessarily monotonic in
  the same direction as GII — the fitted coefficient's sign should be
  interpreted case by case, not assumed to match GII's.

EXPLICIT LIMITATION: gap_score is relative to that year's country sample,
with a single explanatory variable. It does not control for GDP, access
to healthcare systems, or armed conflict — all factors that affect both
measured gender inequality and maternal mortality. It is not a causal
measure.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def compute_gap_score(
    df: pd.DataFrame,
    outcome_col: str = "maternal_mortality_ratio",
    indicator_col: str = "gii",
) -> pd.DataFrame:
    """
    Adds a 'gap_score' column computed over
    log(outcome_col) vs. indicator_col.
    """
    df = df.copy()
    valid_mask = (
        df[outcome_col].notna()
        & df[indicator_col].notna()
        & (df[outcome_col] > 0)
    )

    if valid_mask.sum() < 10:
        raise ValueError(
            "Fewer than 10 valid observations: not enough to fit a "
            "reliable regression. Check the year/region/indicator filter."
        )

    log_outcome = np.log(df.loc[valid_mask, outcome_col]).values
    indicator = df.loc[valid_mask, indicator_col].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(indicator, log_outcome)
    predicted_log = model.predict(indicator)

    df["gap_score"] = np.nan
    df.loc[valid_mask, "gap_score"] = log_outcome - predicted_log
    df.loc[valid_mask, f"{outcome_col}_predicted_log"] = predicted_log

    return df
