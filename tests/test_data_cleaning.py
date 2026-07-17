"""
Unit tests for data_cleaning.py, metrics.py and clustering.py.
Use synthetic data — no network calls, run in any environment.
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import clean_and_merge
from src.metrics import compute_gap_score
from src.clustering import run_clustering


def _make_raw_fixture():
    """Synthetic CSVs mimicking the real shape of OWID data: GII with no
    region column, GDI and maternal mortality with region (empty for
    aggregates)."""
    mmr = pd.DataFrame({
        "Entity": ["Alpha", "Alpha", "Beta", "Gamma", "World", "Arab States (UNDP)"],
        "Code": ["ALP", "ALP", "BET", "GAM", "OWID_WRL", None],
        "Year": [2018, 2019, 2019, 2019, 2019, 2019],
        "maternal_mortality_ratio": [50.0, 45.0, 200.0, 800.0, 150.0, 300.0],
        "region": ["Europe", "Europe", "Europe", "Africa", None, None],
    })

    gii = pd.DataFrame({
        "Entity": ["Alpha", "Alpha", "Beta", "Gamma"],
        "Code": ["ALP", "ALP", "BET", "GAM"],
        "Year": [2018, 2019, 2019, 2019],
        "gii": [0.1, 0.09, 0.3, 0.65],
    })

    gdi = pd.DataFrame({
        "Entity": ["Alpha", "Beta"],
        "Code": ["ALP", "BET"],
        "Year": [2019, 2019],
        "gdi": [0.98, 0.95],
    })

    return {"maternal_mortality": mmr, "gii": gii, "gdi": gdi}


def test_clean_and_merge_filters_aggregates():
    raw = _make_raw_fixture()
    merged = clean_and_merge(raw, save=False)

    assert "World" not in merged["Entity"].values
    assert "Arab States (UNDP)" not in merged["Entity"].values
    assert set(merged["Entity"].unique()) == {"Alpha", "Beta", "Gamma"}


def test_clean_and_merge_keeps_missing_indicators_as_nan():
    raw = _make_raw_fixture()
    merged = clean_and_merge(raw, save=False)

    gamma_row = merged[merged["Entity"] == "Gamma"]
    # Gamma has no GDI in the fixture; it should stay NaN, not be excluded.
    assert gamma_row["gdi"].isna().all()
    assert gamma_row["gii"].notna().all()


def test_compute_gap_score_outlier_detection():
    np.random.seed(0)
    n = 30
    gii = np.random.uniform(0.02, 0.8, n)
    mmr = np.exp(2 + 6 * gii + np.random.normal(0, 0.3, n))
    df = pd.DataFrame({
        "Entity": [f"C{i}" for i in range(n)],
        "gii": gii,
        "maternal_mortality_ratio": mmr,
    })
    df.loc[len(df)] = ["Outlier", 0.05, 400.0]  # low GII, high mortality

    scored = compute_gap_score(df, outcome_col="maternal_mortality_ratio", indicator_col="gii")
    outlier_score = scored.loc[scored["Entity"] == "Outlier", "gap_score"].iloc[0]
    median_score = scored["gap_score"].median()

    assert outlier_score > median_score + 1, "The outlier should have a gap_score clearly above the median."


def test_run_clustering_produces_valid_labels():
    np.random.seed(1)
    n = 40
    gii = np.random.uniform(0.02, 0.8, n)
    mmr = np.exp(2 + 6 * gii + np.random.normal(0, 0.3, n))
    df = pd.DataFrame({
        "Entity": [f"C{i}" for i in range(n)],
        "gii": gii,
        "maternal_mortality_ratio": mmr,
    })

    clustered, meta = run_clustering(df, outcome_col="maternal_mortality_ratio", indicator_col="gii")

    assert "cluster" in clustered.columns
    assert clustered["cluster"].notna().all()
    assert meta["k"] >= 2
