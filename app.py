"""
app.py
------
Interactive dashboard: gender inequality and maternal mortality.

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_cleaning import clean_and_merge, get_snapshot
from src.metrics import compute_gap_score
from src.clustering import run_clustering

st.set_page_config(
    page_title="Gender Inequality and Maternal Mortality",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_HIGHLIGHT = ["Afghanistan", "Norway", "Rwanda", "Bolivia"]


@st.cache_data(ttl="24h", show_spinner="Downloading and cleaning OWID data...")
def get_data() -> pd.DataFrame:
    return clean_and_merge()


try:
    df = get_data()
except Exception as e:
    st.error(
        f"Could not load data from Our World in Data. "
        f"Check your internet connection. Error detail: {e}"
    )
    st.stop()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Controls")

indicator_label_map = {"GII (Gender Inequality Index)": "gii", "GDI (Gender Development Index)": "gdi"}
indicator_label = st.sidebar.radio(
    "Gender inequality indicator",
    options=list(indicator_label_map.keys()),
    help=(
        "GII: 0 = parity, 1 = high inequality. Measures structural "
        "inequality in reproductive health, empowerment, and the labor "
        "market. "
        "GDI: ~1 = parity, values >1 mean women exceed men in human "
        "development. It's a ratio, not a structural inequality measure — "
        "interpret it with caution."
    ),
)
indicator_col = indicator_label_map[indicator_label]

available_years = sorted(df["Year"].dropna().unique().astype(int))
year_mode = st.sidebar.radio("Time mode", ["Single year (snapshot)", "Explore time series"])

if year_mode == "Single year (snapshot)":
    try:
        default_snapshot_df = get_snapshot(df, "maternal_mortality_ratio", indicator_col)
        default_year = int(default_snapshot_df["Year"].iloc[0]) if not default_snapshot_df.empty else available_years[-1]
    except ValueError:
        default_year = available_years[-1]
    selected_year = st.sidebar.selectbox(
        "Year", options=available_years, index=available_years.index(default_year)
    )
else:
    selected_year = st.sidebar.select_slider("Year", options=available_years, value=available_years[-1])

regions_available = sorted(df["region"].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "Filter by region", options=regions_available, default=regions_available
)

highlight_countries = st.sidebar.multiselect(
    "Countries to highlight",
    options=sorted(df["Entity"].unique()),
    default=[c for c in DEFAULT_HIGHLIGHT if c in df["Entity"].unique()],
)

# --------------------------------------------------------------------------
# Filtering
# --------------------------------------------------------------------------

year_df = df[(df["Year"] == selected_year) & (df["region"].isin(selected_regions))].copy()
year_df = year_df.dropna(subset=["maternal_mortality_ratio", indicator_col])

st.title("Gender Inequality and Maternal Mortality")
st.caption(
    "Source: Our World in Data — Gender Inequality Index / Gender Development Index "
    "(UNDP Human Development Report), Maternal Mortality Ratio (UN MMEIG, WHO)."
)

if year_df.empty:
    st.warning(
        f"Not enough data for {int(selected_year)} with the current filters "
        f"({indicator_label}). Try another year or broaden the regions."
    )
    st.stop()

n_countries = year_df["Entity"].nunique()
st.info(f"Showing **{n_countries} countries** for **{int(selected_year)}** — indicator: **{indicator_label}**.")

# --------------------------------------------------------------------------
# Section 1: Interactive exploration
# --------------------------------------------------------------------------

st.header("1. Gender inequality vs. maternal mortality")

year_df["is_highlighted"] = year_df["Entity"].isin(highlight_countries)

fig_scatter = px.scatter(
    year_df,
    x=indicator_col,
    y="maternal_mortality_ratio",
    color="region",
    size="maternal_mortality_ratio",
    hover_name="Entity",
    log_y=True,
    labels={
        indicator_col: indicator_label,
        "maternal_mortality_ratio": "Maternal mortality (per 100k live births, log scale)",
        "region": "Region",
    },
    title=f"Maternal mortality vs. {indicator_label} ({int(selected_year)})",
)

highlighted = year_df[year_df["is_highlighted"]]
if not highlighted.empty:
    fig_scatter.add_trace(
        go.Scatter(
            x=highlighted[indicator_col],
            y=highlighted["maternal_mortality_ratio"],
            mode="markers+text",
            marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(width=3, color="black")),
            text=highlighted["Entity"],
            textposition="top center",
            name="Highlighted countries",
            showlegend=True,
        )
    )

fig_scatter.update_layout(height=560)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown(
    "**How to read this chart:** the Y axis is on a log scale because "
    "maternal mortality varies by orders of magnitude between countries "
    "(from under 5 to over 1000 deaths per 100,000 live births). A "
    "country sitting well above its region's general trend has worse "
    "outcomes than its gender inequality level would predict — and vice "
    "versa."
)

# --------------------------------------------------------------------------
# Section 2: Trend and gap score
# --------------------------------------------------------------------------

st.header("2. The gap between expected inequality and observed outcome")

year_df_scored = compute_gap_score(year_df, outcome_col="maternal_mortality_ratio", indicator_col=indicator_col)

fig_trend = px.scatter(
    year_df_scored,
    x=indicator_col,
    y="maternal_mortality_ratio",
    trendline="ols",
    trendline_options=dict(log_y=True),
    log_y=True,
    hover_name="Entity",
    labels={
        indicator_col: indicator_label,
        "maternal_mortality_ratio": "Maternal mortality (log scale)",
    },
    title=f"Log-linear fit: log(maternal mortality) ~ {indicator_label}",
)
fig_trend.update_layout(height=480)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown(
    f"""
The log-linear fit shows the typical relationship between {indicator_label} and
maternal mortality in the {int(selected_year)} sample. Each country's **gap_score**
is the residual of that regression: how much more (or less) maternal mortality a
country has than its gender inequality level would predict, **relative to this
sample**.

**Note on direction:** if you're using GII, a positive relationship is expected
(more inequality -> more mortality). If you switch to GDI in the sidebar, the
relationship's direction may not be the same — GDI is a male/female human
development ratio, not a structural inequality measure, and its relationship
with maternal mortality is empirical, not assumed.
"""
)

# --------------------------------------------------------------------------
# Section 3: Clustering
# --------------------------------------------------------------------------

st.header("3. Country profile clusters (K-means)")

with st.expander("How is this calculated?", expanded=False):
    st.markdown(
        f"""
Countries are grouped by two standardized variables: `{indicator_label}` and
`log(maternal mortality)`. The number of clusters (k) is chosen automatically
by maximizing the *silhouette score* between k=2 and k=6.

**Label limitation:** each cluster's label is generated by comparing its
median against the global median (high/low). With k > 2, two distinct
clusters can end up with the same label — in that case, check each cluster's
size and country list to tell them apart, don't assume they're the same
group.
"""
    )

try:
    clustered_df, cluster_meta = run_clustering(
        year_df, outcome_col="maternal_mortality_ratio", indicator_col=indicator_col
    )
except ValueError as e:
    st.warning(f"Could not run clustering: {e}")
    clustered_df, cluster_meta = None, None

if clustered_df is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        fig_cluster = px.scatter(
            clustered_df.dropna(subset=["cluster_label"]),
            x=indicator_col,
            y="maternal_mortality_ratio",
            color="cluster_label",
            hover_name="Entity",
            log_y=True,
            labels={
                indicator_col: indicator_label,
                "maternal_mortality_ratio": "Maternal mortality (log scale)",
                "cluster_label": "Profile",
            },
            title=f"Clusters (k={cluster_meta['k']}, chosen by silhouette score)",
        )
        fig_cluster.update_layout(height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col2:
        st.markdown("**Silhouette score per k tested:**")
        sil_df = pd.DataFrame(
            {"k": list(cluster_meta["silhouette_scores"].keys()),
             "silhouette": list(cluster_meta["silhouette_scores"].values())}
        )
        st.dataframe(sil_df, hide_index=True, use_container_width=True)
        st.markdown(f"**k chosen: {cluster_meta['k']}**")

    st.markdown("**Composition of each cluster:**")
    for cluster_id, label in cluster_meta["label_map"].items():
        countries_in_cluster = clustered_df.loc[
            clustered_df["cluster"] == cluster_id, "Entity"
        ].tolist()
        preview = ", ".join(countries_in_cluster[:8])
        suffix = "..." if len(countries_in_cluster) > 8 else ""
        st.markdown(f"- **{label}** ({len(countries_in_cluster)} countries): {preview}{suffix}")

# --------------------------------------------------------------------------
# Section 4: Outliers
# --------------------------------------------------------------------------

st.header("4. Outliers: worse and better than expected")

top_worse = year_df_scored.dropna(subset=["gap_score"]).nlargest(10, "gap_score")
top_better = year_df_scored.dropna(subset=["gap_score"]).nsmallest(10, "gap_score")

col_a, col_b = st.columns(2)
with col_a:
    st.subheader(f"Worse maternal mortality than expected from their {indicator_label}")
    st.dataframe(
        top_worse[["Entity", indicator_col, "maternal_mortality_ratio", "gap_score"]]
        .rename(columns={
            "Entity": "Country", indicator_col: indicator_label,
            "maternal_mortality_ratio": "Maternal mortality", "gap_score": "Score",
        }).round(2),
        hide_index=True, use_container_width=True,
    )
with col_b:
    st.subheader(f"Better maternal mortality than expected from their {indicator_label}")
    st.dataframe(
        top_better[["Entity", indicator_col, "maternal_mortality_ratio", "gap_score"]]
        .rename(columns={
            "Entity": "Country", indicator_col: indicator_label,
            "maternal_mortality_ratio": "Maternal mortality", "gap_score": "Score",
        }).round(2),
        hide_index=True, use_container_width=True,
    )

# --------------------------------------------------------------------------
# Methodological note
# --------------------------------------------------------------------------

st.divider()
st.subheader("⚠️ Methodological note")
st.markdown(
    """
- **Correlation does not imply causation.** This analysis shows observed
  associations between gender inequality and maternal mortality, not
  causal relationships. Factors such as GDP per capita, access to
  healthcare systems, and armed conflict affect both variables
  simultaneously.
- **GII and GDI measure different things.** GII is a structural inequality
  measure (0=parity, 1=high inequality). GDI is a human development ratio
  between women and men (~1=parity). They are not interchangeable — the
  sidebar toggle lets you compare both perspectives, it doesn't assume
  they measure the same thing.
- **The "gap_score" and clusters are descriptive**, computed on only two
  observed variables. They do not control for other determinants of
  maternal mortality.
- **Uneven coverage**: not every country reports GII/GDI for every year in
  which maternal mortality data exists — those rows remain NaN and are
  excluded from the analysis for that year, which can introduce bias if
  the missingness isn't random.
- Data: [Our World in Data](https://ourworldindata.org/) — Gender
  Inequality Index and Gender Development Index (UNDP Human Development
  Report), Maternal Mortality Ratio (UN MMEIG, WHO Mortality Database).
"""
)
