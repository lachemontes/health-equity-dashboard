"""
data_loader.py
---------------
Downloads (with local caching) the three Our World in Data datasets used in
this dashboard:
- Gender Inequality Index (GII)
- Gender Development Index (GDI)
- Maternal mortality ratio

Design (same pattern as the healthcare-expenditure dashboard, reused
because it's already proven):
- Caches each raw CSV in data/raw/ to avoid re-downloading on every
  Streamlit rerun.
- Dynamically detects the "value" column name instead of hardcoding it,
  because each OWID dataset uses a different, verbose column name
  (e.g. "Gender Inequality Index").

KEY DIFFERENCE from the previous dashboard: not every dataset here ships a
region column. GII doesn't have one; GDI and maternal-mortality do
("World region according to OWID"). _standardize_columns() detects the
region column if present and leaves it out otherwise — the aggregate-
filtering logic (in data_cleaning.py) uses maternal-mortality's region as
the authoritative source, not each dataset's own.
"""

from pathlib import Path
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Our World In Data data fetch/1.0"}

DATASETS = {
    "gii": {
        "url": (
            "https://ourworldindata.org/grapher/"
            "gender-inequality-index-from-the-human-development-report.csv"
            "?v=1&csvType=full&useColumnShortNames=false"
        ),
        "value_col": "gii",
        "filename": "gii.csv",
    },
    "gdi": {
        "url": (
            "https://ourworldindata.org/grapher/"
            "gender-development-index.csv"
            "?v=1&csvType=full&useColumnShortNames=false"
        ),
        "value_col": "gdi",
        "filename": "gdi.csv",
    },
    "maternal_mortality": {
        "url": (
            "https://ourworldindata.org/grapher/"
            "maternal-mortality.csv"
            "?v=1&csvType=full&useColumnShortNames=false"
        ),
        "value_col": "maternal_mortality_ratio",
        "filename": "maternal_mortality.csv",
    },
}

ID_COLS = {"Entity", "Code", "Year"}
REGION_COL_RAW = "World region according to OWID"


def _download_csv(url: str, dest_path: Path) -> None:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    dest_path.write_bytes(response.content)


def _standardize_columns(df: pd.DataFrame, value_col_name: str) -> pd.DataFrame:
    """
    Renames the value column to a short name, and the region column
    (if present) to 'region'. Drops text annotation columns (e.g.
    "Maternal mortality ratio (Annotations)") that add nothing to the
    analysis.
    """
    df = df.copy()

    annotation_cols = [c for c in df.columns if "(Annotations)" in c or "(annotations)" in c]
    df = df.drop(columns=annotation_cols, errors="ignore")

    has_region = REGION_COL_RAW in df.columns
    if has_region:
        df = df.rename(columns={REGION_COL_RAW: "region"})

    value_cols = [c for c in df.columns if c not in ID_COLS and c != "region"]
    if len(value_cols) != 1:
        raise ValueError(
            f"Expected exactly 1 value column, found {len(value_cols)}: "
            f"{value_cols}. Inspect the CSV manually."
        )
    return df.rename(columns={value_cols[0]: value_col_name})


def load_dataset(key: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Loads a dataset by key ('gii', 'gdi', 'maternal_mortality'),
    downloading it if it's not cached or if force_refresh=True.
    """
    if key not in DATASETS:
        raise KeyError(f"Unknown dataset: {key}. Options: {list(DATASETS)}")

    config = DATASETS[key]
    dest_path = RAW_DIR / config["filename"]

    if force_refresh or not dest_path.exists():
        _download_csv(config["url"], dest_path)

    df = pd.read_csv(dest_path)
    df = _standardize_columns(df, config["value_col"])
    return df


def load_all_raw(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {key: load_dataset(key, force_refresh) for key in DATASETS}
