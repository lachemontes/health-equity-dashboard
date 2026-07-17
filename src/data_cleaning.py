"""
data_cleaning.py
-----------------
Cleaning and merging of GII, GDI and maternal mortality.

Design decisions:

1. REGION SOURCE: maternal-mortality.csv ships the region column
   ("World region according to OWID"). GII doesn't. Instead of
   downloading a 4th dataset just for the country->region mapping (as in
   the healthcare-expenditure dashboard), maternal mortality's region is
   used as the authoritative source — it's the same OWID convention, so
   country names match exactly.

2. FILTERING AGGREGATES: in OWID CSVs, aggregate entities (e.g. "World",
   "Arab States (UNDP)", "Africa") have an EMPTY region column — by OWID
   design, region is only assigned to real countries. Filtering by
   `region.notna()` excludes aggregates without needing a hardcoded
   exclusion list.

3. MERGE: we start from maternal mortality (our outcome variable, the
   most complete in coverage) and LEFT JOIN GII and GDI onto it. This
   preserves every maternal-mortality row even if GII or GDI is missing
   for that country-year — those become NaN, visible and filterable in
   the dashboard, not hidden. The alternative (strict inner join) would
   shrink the sample to the intersection of all three sources, dropping
   valid country-years just because ONE of the two inequality metrics is
   missing.
"""

from pathlib import Path

import pandas as pd

from src.data_loader import load_all_raw

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def clean_and_merge(
    raw: dict[str, pd.DataFrame] | None = None, save: bool = True
) -> pd.DataFrame:
    """
    Combines GII, GDI and maternal mortality into a single long-format
    DataFrame.

    Returns columns:
        Entity, Code, Year, maternal_mortality_ratio, region, gii, gdi
    """
    if raw is None:
        raw = load_all_raw()

    mmr = raw["maternal_mortality"]
    gii = raw["gii"][["Entity", "Code", "Year", "gii"]]
    gdi = raw["gdi"][["Entity", "Code", "Year", "gdi"]]

    # Filter aggregates: empty region marks non-country entities in OWID
    mmr = mmr[mmr["region"].notna()].copy()

    n_before = mmr["Code"].nunique()

    merged = mmr.merge(gii, on=["Entity", "Code", "Year"], how="left")
    merged = merged.merge(gdi, on=["Entity", "Code", "Year"], how="left")

    n_with_gii = merged.loc[merged["gii"].notna(), "Code"].nunique()
    n_with_gdi = merged.loc[merged["gdi"].notna(), "Code"].nunique()
    print(
        f"[data_cleaning] Countries with maternal mortality data: {n_before}. "
        f"Of those, with GII available for at least one year: {n_with_gii}, "
        f"with GDI available for at least one year: {n_with_gdi}."
    )

    result = merged.sort_values(["Entity", "Year"]).reset_index(drop=True)

    if save:
        dest = PROCESSED_DIR / "merged_dataset.csv"
        result.to_csv(dest, index=False)
        print(f"[data_cleaning] Cleaned dataset saved to {dest}")

    return result


def get_snapshot(
    df: pd.DataFrame, outcome_col: str, indicator_col: str, year: int | None = None
) -> pd.DataFrame:
    """
    Returns the snapshot for a specific year for the selected
    (outcome, inequality indicator) pair. If year=None, uses the most
    recent year with reasonable coverage (>= 30 countries with both
    variables non-null).
    """
    valid = df.dropna(subset=[outcome_col, indicator_col])

    if year is not None:
        return valid[valid["Year"] == year]

    counts_by_year = valid.groupby("Year")["Code"].nunique().sort_index()
    valid_years = counts_by_year[counts_by_year >= 30]
    if valid_years.empty:
        raise ValueError("No year meets the minimum coverage of 30 countries for this variable pair.")
    latest_valid_year = valid_years.index.max()
    return valid[valid["Year"] == latest_valid_year]
