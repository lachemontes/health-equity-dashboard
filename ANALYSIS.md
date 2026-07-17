# How this analysis was built

This document walks through the reasoning behind each design decision in
this project — not just what the code does, but why I made each choice.
It's meant as a companion to the code for anyone (including future me)
who wants to understand or defend these decisions.

## 1. The question

Does gender inequality help explain differences in maternal mortality
between countries? I wanted a dashboard that lets someone explore that
relationship interactively, not just show a static correlation number.

## 2. Choosing the data sources

I started from Our World in Data's Gender Inequality Index (GII), assuming
it would break down into sub-components like maternal mortality. It
doesn't — the exported CSV gives a single composite score per
country-year. GII *incorporates* maternal mortality and adolescent birth
rate into its formula, but doesn't expose them separately.

That meant I needed a second, independent dataset for the health outcome:
OWID's `maternal-mortality` dataset (sourced from UN MMEIG / WHO). This
matters for the analysis: GII and maternal mortality are not fully
independent by construction (GII's formula already includes a maternal
mortality component), so the relationship I'm measuring is partly
circular by design. I call this out explicitly in the dashboard's
methodological note rather than presenting it as a clean, independent
correlation.

I also added GDI (Gender Development Index) as a secondary, toggleable
indicator — but GDI measures something different: a ratio of human
development between women and men, not a structural inequality score.
I kept them as separate toggle options rather than treating them as
interchangeable, because conflating them would misrepresent what each one
measures.

## 3. Filtering out non-country entities

OWID's CSVs mix real countries with aggregates ("World", "Arab States
(UNDP)", "Africa", income-group aggregates, etc.). Including those in the
analysis would double-count data (a country's data point plus its
region's aggregate) and break per-country comparisons.

My first instinct was to maintain a manual exclusion list. That's fragile
— it breaks silently if OWID adds a new aggregate category later. Instead,
I noticed that OWID's own `region` column is **empty** for every aggregate
row and populated for every real country, by their own data convention.
Filtering with `region.notna()` gets the same result with one line, and
it stays correct even if OWID adds new aggregate types in the future,
because the convention (not a name I hardcoded) is what does the
filtering.

## 4. The merge strategy: why left join, not inner join

I merge starting from maternal mortality (the most complete source) and
left-join GII and GDI onto it, rather than inner-joining all three.

The difference matters: an inner join would only keep country-years where
*all three* variables are present, silently dropping any country-year
missing just one indicator. A left join keeps every maternal-mortality
row and lets GII/GDI be `NaN` when unavailable — visible in the data,
filterable in the dashboard, not silently discarded. This preserves more
of the sample and makes missingness an explicit, inspectable fact rather
than an invisible one.

## 5. Why log-transform maternal mortality

Maternal mortality ranges from under 5 deaths per 100,000 live births
(Nordic countries) to over 1,500 (some historical low-income-country
values). That's a three-order-of-magnitude spread. A regression on the
raw scale would be dominated by the handful of extreme values — the fit
line would barely respond to variation among the low-mortality majority
of countries.

Taking `log(maternal_mortality)` compresses that range and makes the
regression sensitive to *proportional* differences rather than absolute
ones, which is also a more meaningful way to think about mortality:
going from 10 to 20 deaths per 100k is the same relative change as going
from 500 to 1000, even though the absolute gap is very different.

## 6. Interpreting the gap score

`gap_score` is the residual of `log(maternal_mortality) ~ indicator`: how
far a country's actual (log) mortality sits above or below what the
regression line predicts for its inequality level.

A few things worth being precise about when explaining this:
- It's **relative to the sample**, not an absolute measure. The same
  country's gap_score can change year to year even if nothing about the
  country changed, just because the composition of the sample (which
  countries have data that year) shifted the regression line.
- It's **not causal**. A country can have a large positive gap_score for
  reasons entirely unrelated to gender inequality — armed conflict,
  healthcare infrastructure collapse, a demographic shock in reported
  data. The score flags a pattern worth investigating, not an
  explanation.
- The **direction of the expected relationship differs between GII and
  GDI**. GII is scaled so that higher always means more inequality, so a
  positive coefficient in the regression is expected. GDI is a ratio
  centered near 1, and its relationship with maternal mortality is not
  guaranteed to run in a single, obvious direction — I don't assume it
  matches GII's pattern, I let the fitted coefficient speak for itself.

## 7. Why K-means with standardized features and silhouette-selected k

I clustered on `[indicator, log(maternal_mortality)]` after standardizing
both (mean 0, standard deviation 1). Without standardizing, K-means would
be dominated by whichever variable has the larger numeric range — in this
case log(mortality) would likely dominate GII, distorting the clusters
toward outcome alone rather than the joint pattern I actually want to
see.

Rather than picking k by hand, I test k=2 through k=6 and pick whichever
maximizes the silhouette score — a measure of how well-separated and
internally cohesive the resulting clusters are. This avoids the temptation
to force a "nice round number" of clusters that doesn't actually reflect
structure in the data.

**A limitation I found through testing, not through reading a textbook:**
my cluster-labeling function compares each cluster's median against the
*global* median (e.g. "high GII, high mortality"). With k=2 this always
produces distinct labels. With k>2, I found in testing that two different
clusters can end up with the *same* label, because both fall on the same
side of the global median even though K-means separated them for another
reason (usually spread within that quadrant). I decided not to build a
more complex cross-cluster labeling scheme to fix this — the labels are
meant as a quick descriptive aid, and the underlying cluster assignment
is still correct even when two labels collide. I documented this
explicitly in the code and the dashboard's UI rather than hiding it.

## 8. What I'd say if asked "why not just correlation?"

A single correlation coefficient (e.g. Pearson's r) would answer "how
strongly are these two variables linearly related," but it wouldn't let
someone explore *which* countries deviate from that pattern, *by how
much*, or *how outliers cluster together*. The regression residual
(gap_score) and clustering give a finer-grained, explorable view of the
same underlying relationship, at the cost of added assumptions (linearity
in the log scale, the specific k chosen for clustering) that I've tried
to make explicit rather than hide behind a single summary number.

## 9. What I'd change with more time

- Add a covariate (e.g. GDP per capita) to the gap_score regression to
  partial out at least one obvious confound, and be explicit that even a
  multi-variable regression wouldn't make the result causal.
- Investigate the circularity issue from section 2 more rigorously —
  ideally by finding whether OWID or UNDP publishes the GII
  sub-components separately, which would let me build a cleaner,
  non-circular comparison.
- Extend the time-series view to show *within-country* trajectories over
  time (how a country's gap_score has moved), not just cross-sectional
  snapshots.
