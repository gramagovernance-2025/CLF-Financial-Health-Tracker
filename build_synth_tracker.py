"""
build_synth_tracker.py — FAKE-DATA PROTOTYPE generator for the new,
standalone "Synthesised CLF Tracker" (8 financial/audit indicators:
OSS, combined ICF+GL OTR, combined ICF+GL UPTO, loan disbursement
amount+%, bank balance, cash balance, idle funds, PAR).

This is explicitly a structure/format prototype, not real data - every
indicator value below is randomly generated. The point is to validate
navigation, layout, choropleth, ranking-table, and weight-customization
mechanics before the real audit/F05/F01 data pipeline is wired in.

Reused, not reinvented:
  - The District -> Block -> CLF hierarchy (real names + MIS IDs) comes
    from the ALREADY-BUILT comprehensive tracker's own manifest
    (../CLF Tracker/Scale-Up/data/manifest.json, 38 districts, 1,667
    CLFs) - not invented from scratch, since a real, complete CLF
    universe already exists in this repo.
  - The 534-block / 38-district choropleth geometry is copied from the
    DAK Dashboard's own build (data/geo/*.geojson, block_universe_534.csv)
    - same block-status/case-heatmap component precedent, reused here.
  - Scoring methodology mirrors this project's own established
    convention (DAK tracker, audit_analysis.do): every raw metric is
    converted to a STATE PERCENTILE (direction-aware: lower_is_better
    for cash/bank balance and idle funds), then averaged into an Overall
    Score - not an invented 0-100 "health formula."

Indicators, direction, and native cadence:
  - oss              (%, higher better, quarterly)
  - otr               (%, higher better, quarterly)  [combined ICF+GL]
  - upto              (%, higher better, quarterly)  [combined ICF+GL]
  - disb_amount       (Rs, magnitude only - NOT in the composite)
  - disb_pct          (%, higher better, quarterly)
  - cash_balance      (Rs, lower better, quarter-end - see below)
  - bank_balance      (Rs, lower better, quarter-end, + 3-tier persistence status)
  - idle_funds        (Rs, lower better, summed over the quarter's 3 months)
  - par = 100 - otr   (%, lower better, quarterly)  [display-only, excluded
                        from the Overall Score composite - redundant with otr]

Every indicator is now QUARTER-NATIVE (see "Quarter-pinning the monthly
balance-sheet metrics" below) - there is no more monthly/quarterly split.
The whole tracker is driven by ONE period toggle (a quarter), and every
section - KPI tiles, category-distribution donuts, ranking tables, the
choropleth - reads off that same selected quarter via a `by_period` dict
on every CLF/block/district/state payload (same convention as the DAK
Tracker's own by_period model in 3_Output/DAK Dashboard). The raw MONTHLY
cash/bank/idle-funds series is still shipped (under `series`, unchanged)
so the frontend can show the 3 individual months within whichever
quarter is selected, and so the "Explore by Place and Period" trend
chart can still plot every month - that chart is deliberately NOT
affected by the quarter toggle, same precedent as the DAK Tracker's own
trend charts.

Quarter-pinning the monthly balance-sheet metrics (agreed with Mohan):
  - Cash Balance's quarter figure = the quarter-END value (last month of
    the quarter) - matches how a balance sheet is normally read.
  - Bank Balance's persistence status is now evaluated WITHIN the
    calendar quarter only (not a rolling window that could cross quarter
    boundaries as the very first prototype pass did): count how many of
    the quarter's 3 months were >= the Rs 25L threshold - 0 -> green,
    1-2 -> yellow, all 3 -> red. The quarter-end value is still what's
    displayed as the headline Bank Balance figure.
  - Idle Funds: cash excess over Rs 25,000 counts EVERY month,
    unconditionally, summed across the quarter's 3 months (Mohan: "cash
    excess in each month count towards idle fund"). Bank excess over
    Rs 25L only counts if the quarter's bank status is red (all 3 months
    over threshold) - in that case all 3 months' bank excess is included;
    otherwise none of it is. This keeps the original "1-2 months high is
    fine, 3+ is a flag" intent, just scoped to the calendar quarter
    instead of an arbitrary trailing window.

Composite ("Overall Score"): equal-weight average of state percentiles
of {oss, otr, upto, disb_pct, cash_balance, bank_balance, idle_funds}
(7 of the 8 - PAR excluded, per explicit user decision), computed
SEPARATELY for each quarter - a CLF's Overall Score and every rank/
percentile can differ quarter to quarter, same as any of the other
composite indicators. Customizable per-indicator weighting is a
CLIENT-SIDE feature (see index.html) - this script only needs to ship
every unit's raw + percentile scores so the frontend can recompute live,
same mechanic as the existing tracker's weight modal
(data/scoring_summary.json equivalent below, also now by-quarter).
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

HERE = Path(__file__).parent
EXISTING_TRACKER = HERE.parent / "CLF Tracker" / "Scale-Up"
OUT = HERE / "data"

QUARTERS = ["FY25-26 Q4", "FY26-27 Q1", "FY26-27 Q2"]
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
# Falls out for free - the 9 scraped months divide evenly into the 3
# FY quarters already used for the audit-derived indicators.
QUARTER_MONTHS = {
    "FY25-26 Q4": MONTHS[0:3],
    "FY26-27 Q1": MONTHS[3:6],
    "FY26-27 Q2": MONTHS[6:9],
}
LATEST_Q = QUARTERS[-1]
# "cumulative" is a 4th, DEFAULT period alongside the 3 real quarters -
# per-indicator meaning (agreed with Mohan, documented where each is
# computed below):
#   oss/otr/disb_pct/par      -> simple average across the 3 quarters
#                                 (no stored numerator/denominator to sum,
#                                 so an average is the best available
#                                 proxy for "how this has looked overall")
#   upto/disb_amount          -> latest quarter's value (both are already
#                                 cumulative-since-inception BY THEIR OWN
#                                 DEFINITION, so "cumulative" = whatever
#                                 the most recent one says, not an average
#                                 of already-cumulative snapshots)
#   cash_balance/bank_balance -> latest quarter's value (a balance is a
#                                 current state, not something you sum
#                                 across periods)
#   idle_funds                -> summed across all 3 quarters (the one
#                                 genuinely additive/flow metric here)
PERIOD_KEYS = ["cumulative"] + QUARTERS
DEFAULT_PERIOD = "cumulative"

CASH_THRESHOLD = 25_000
BANK_THRESHOLD = 25_00_000

# ---------------------------------------------------------------- category-bucket thresholds
# Per explicit user instruction (OSS 2-group at 100%, cash/bank 3-tier,
# PAR 2-group at 2%) plus provisional thresholds proposed for the
# remaining rate indicators - flagged to the user as adjustable.
def oss_bucket(v):
    if v >= 100: return "green"
    if v >= 75: return "yellow"
    return "red"

def par_bucket(v):
    # lower PAR is better, so below the 2% threshold is green
    if v < 2: return "green"
    if v <= 5: return "yellow"
    return "red"

def otr_bucket(v):
    if v >= 95: return "green"
    if v >= 75: return "yellow"
    return "red"

def upto_bucket(v):
    if v >= 90: return "green"
    if v >= 60: return "yellow"
    return "red"

def disb_pct_bucket(v):
    if v >= 90: return "green"
    if v >= 60: return "yellow"
    return "red"

def idle_funds_bucket(v):
    if v <= 0: return "green"
    if v <= 5_00_000: return "yellow"
    return "red"

def cash_bucket(v):
    # magnitude band around the Rs 25k threshold (not persistence-based,
    # unlike bank balance) - matches what was agreed for cash balance.
    if v < 20_000: return "green"
    if v < 30_000: return "yellow"
    return "red"

BUCKET_FN = {
    "oss": oss_bucket, "par": par_bucket, "otr": otr_bucket, "upto": upto_bucket,
    "disb_pct": disb_pct_bucket, "idle_funds": idle_funds_bucket, "cash_balance": cash_bucket,
}
BUCKET_INDICATORS = list(BUCKET_FN.keys()) + ["bank_balance"]  # bank_balance uses its own persistence status, not a BUCKET_FN
BUCKET_LABELS_2GROUP = set()  # every bucket-eligible indicator is now 3-group (green/yellow/red)

# ---------------------------------------------------------------- load the real hierarchy
manifest = json.load(open(EXISTING_TRACKER / "data" / "manifest.json"))
clf_rows = []
for d in manifest["districts"]:
    for b in d["blocks"]:
        for c in b["clfs"]:
            clf_rows.append({"mis_id": c["id"], "clf_name": c["n"], "district": d["name"], "block": b["name"]})
clf_df = pd.DataFrame(clf_rows)
print(f"Loaded real hierarchy: {clf_df['district'].nunique()} districts, "
      f"{clf_df.groupby(['district','block']).ngroups} blocks, {len(clf_df)} CLFs")

# ---------------------------------------------------------------- geo id matching (best-effort, prototype-grade)
def norm_block(b):
    import re
    b = str(b).strip().upper()
    b = re.sub(r"\bSADAR\b", "", b)
    b = re.sub(r"[.\-]", " ", b)
    b = re.sub(r"\s+", " ", b).strip()
    return b

DISTRICT_ALIAS = {
    "EAST CHAMPARAN": "PURBI CHAMPARAN", "WEST CHAMPARAN": "PASHCHIM CHAMPARAN",
    "KAIMUR": "KAIMUR (BHABUA)", "SARHASA": "SAHARSA",
}
block_universe = pd.read_csv(HERE / "block_universe_534.csv")
clf_df["district_norm"] = clf_df["district"].str.strip().str.upper().map(lambda x: DISTRICT_ALIAS.get(x, x))
clf_df["block_norm"] = clf_df["block"].apply(norm_block)
_bmap = block_universe.set_index(["district_norm", "block_norm"])["block_id"]
clf_df["block_id"] = clf_df.apply(lambda r: _bmap.get((r["district_norm"], r["block_norm"])), axis=1)
_matched = clf_df.drop_duplicates(["district", "block"])["block_id"].notna().sum()
_total_b = clf_df.drop_duplicates(["district", "block"]).shape[0]
print(f"Block geo-match (prototype-grade, best-effort): {_matched}/{_total_b} blocks matched to the 534-block map "
      f"- unmatched blocks just won't render on the choropleth for this prototype pass.")

# ---------------------------------------------------------------- fake per-CLF generation
def clip(a, lo, hi):
    return np.clip(a, lo, hi)


def gen_clf_series(n):
    """One row of fake values per CLF for every period of every indicator,
    with a deliberately-constructed persistence PROFILE for bank balance
    (green/yellow/red) so the 3-tier flag logic has real cases to show,
    not just noise."""
    out = {}

    # Quarterly indicators
    out["oss"] = {q: clip(np.random.normal(68, 16, n), 5, 140).round(1) for q in QUARTERS}
    out["otr"] = {q: clip(np.random.normal(88, 42, n), 0, 320).round(1) for q in QUARTERS}
    out["upto"] = {q: clip(np.random.normal(64, 28, n), 0, 260).round(1) for q in QUARTERS}
    out["disb_pct"] = {q: clip(np.random.normal(74, 21, n), 0, 118).round(1) for q in QUARTERS}
    out["disb_amount"] = {q: np.random.lognormal(15.2, 1.0, n).round(0) for q in QUARTERS}
    out["par"] = {q: (100 - out["otr"][q]).round(1) for q in QUARTERS}

    # Monthly: cash balance - mostly small, a tail above the 25k line
    out["cash_balance"] = {}
    base_cash = np.random.lognormal(9.0, 1.1, n)
    for m in MONTHS:
        noise = np.random.normal(1.0, 0.15, n)
        out["cash_balance"][m] = clip(base_cash * noise, 0, None).round(0)

    # Monthly: bank balance, built from an explicit persistence profile so
    # green/yellow/red all have real, reviewable cases WITHIN at least one
    # quarter (the profile drives the LAST quarter's status most clearly;
    # earlier quarters are mostly noise around the base level).
    profile = np.random.choice(["green", "yellow", "red"], size=n, p=[0.55, 0.25, 0.20])
    base_level = np.where(
        profile == "green", np.random.lognormal(13.8, 0.9, n),      # mostly well under 25L
        np.where(profile == "yellow", np.random.lognormal(14.9, 0.5, n),  # hovering near/just over
                 np.random.lognormal(15.4, 0.6, n))                  # red: well over, persistently
    )
    out["bank_balance"] = {}
    n_months = len(MONTHS)
    for mi, m in enumerate(MONTHS):
        vals = base_level * np.random.normal(1.0, 0.12, n)
        # yellow: only the LAST 1-2 months spike above threshold (recent
        # deposit story, within the final quarter); red: last 3 months
        # (i.e. the whole final quarter) held above threshold.
        recent_idx = n_months - mi  # months remaining incl. this one, counting from the end
        yellow_mask = (profile == "yellow") & (recent_idx <= 2)
        red_mask = (profile == "red") & (recent_idx <= 3)
        vals = np.where(yellow_mask, np.maximum(vals, BANK_THRESHOLD * np.random.uniform(1.05, 1.4, n)), vals)
        vals = np.where(red_mask, np.maximum(vals, BANK_THRESHOLD * np.random.uniform(1.1, 2.2, n)), vals)
        out["bank_balance"][m] = clip(vals, 0, None).round(0)

    return out, profile


clf_series, clf_bank_profile = gen_clf_series(len(clf_df))


def bank_status_for_quarter(vals_by_month, q):
    """How many of THIS quarter's 3 months were >= the bank threshold -
    0 -> green, 1-2 -> yellow, all 3 -> red. Deliberately scoped to the
    calendar quarter (not a rolling trailing window that could cross
    quarter boundaries) - see module docstring."""
    months = QUARTER_MONTHS[q]
    n_over = sum(1 for m in months if vals_by_month[m] >= BANK_THRESHOLD)
    if n_over >= 3: return "red"
    if n_over >= 1: return "yellow"
    return "green"


def idle_funds_for_month(cash_val, bank_val, quarter_bank_status):
    """Cash excess counts every month, unconditionally. Bank excess only
    counts if the WHOLE quarter this month belongs to is flagged red."""
    idle_cash = max(0.0, cash_val - CASH_THRESHOLD)
    idle_bank = max(0.0, bank_val - BANK_THRESHOLD) if quarter_bank_status == "red" else 0.0
    return round(idle_cash + idle_bank, 0)


MONTH_TO_QUARTER = {m: q for q in QUARTERS for m in QUARTER_MONTHS[q]}

# ---------------------------------------------------------------- assemble per-CLF records
records = []
for i, row in clf_df.reset_index(drop=True).iterrows():
    rec = {"mis_id": int(row["mis_id"]), "clf_name": row["clf_name"], "district": row["district"],
           "block": row["block"], "block_id": row["block_id"] if pd.notna(row["block_id"]) else None}
    for ind in ["oss", "otr", "upto", "disb_pct", "disb_amount", "par"]:
        rec[ind] = {q: float(clf_series[ind][q][i]) for q in QUARTERS}
    rec["cash_balance"] = {m: float(clf_series["cash_balance"][m][i]) for m in MONTHS}
    rec["bank_balance"] = {m: float(clf_series["bank_balance"][m][i]) for m in MONTHS}

    rec["bank_status_q"] = {q: bank_status_for_quarter(rec["bank_balance"], q) for q in QUARTERS}

    idle_funds = {}
    for m in MONTHS:
        idle_funds[m] = idle_funds_for_month(rec["cash_balance"][m], rec["bank_balance"][m],
                                              rec["bank_status_q"][MONTH_TO_QUARTER[m]])
    rec["idle_funds"] = idle_funds

    # Quarter-level rollups that feed scoring/ranking/choropleth - the
    # "one number per quarter" every other indicator already has.
    rec["cash_balance_q"] = {q: rec["cash_balance"][QUARTER_MONTHS[q][-1]] for q in QUARTERS}     # quarter-end
    rec["bank_balance_q"] = {q: rec["bank_balance"][QUARTER_MONTHS[q][-1]] for q in QUARTERS}     # quarter-end
    rec["idle_funds_q"] = {q: round(sum(idle_funds[m] for m in QUARTER_MONTHS[q]), 0) for q in QUARTERS}  # summed

    records.append(rec)

clf_full = pd.DataFrame(records)
print(f"Generated fake series for {len(clf_full)} CLFs.")

# ---------------------------------------------------------------- percentile helper (state-precedent convention)
def pctl(series, lower_is_better=False):
    s = series.astype(float)
    ranks = s.rank(pct=True, ascending=not lower_is_better)
    return (ranks * 100).round(1)


COMPOSITE_INDICATORS = ["oss", "otr", "upto", "disb_pct", "cash_balance", "bank_balance", "idle_funds"]
LOWER_BETTER = {"cash_balance", "bank_balance", "idle_funds", "par"}


def add_group_rank(df, group_col, prefix):
    g = df.groupby(group_col)["overall_score"]
    df[f"overall_{prefix}_rank"] = g.rank(ascending=False, method="min").astype(int)
    df[f"overall_{prefix}_n"] = g.transform("count").astype(int)
    for ind in COMPOSITE_INDICATORS + ["par"]:
        gg = df.groupby(group_col)[ind]
        lb = ind in LOWER_BETTER
        df[f"{ind}_{prefix}_rank"] = gg.rank(ascending=lb, method="min").astype(int)
        df[f"{ind}_{prefix}_n"] = gg.transform("count").astype(int)


def build_flat_for_quarter(q):
    """CLF-level table for ONE period - either a real quarter (every
    indicator drawn straight from that quarter) or "cumulative" (each
    indicator collapsed across all 3 quarters per the rule documented
    next to PERIOD_KEYS above: average for the rate indicators, latest-
    quarter for the already-cumulative ones, sum for idle funds)."""
    if q == "cumulative":
        avg3 = lambda col: clf_full[col].apply(lambda d: float(np.mean([d[qq] for qq in QUARTERS])))
        latest = lambda col: clf_full[col].apply(lambda d: d[LATEST_Q])
        sum3 = lambda col: clf_full[col].apply(lambda d: float(sum(d[qq] for qq in QUARTERS)))
        cols = {
            "oss": avg3("oss"), "otr": avg3("otr"), "upto": latest("upto"),
            "disb_pct": avg3("disb_pct"), "disb_amount": latest("disb_amount"), "par": avg3("par"),
            "cash_balance": latest("cash_balance_q"), "bank_balance": latest("bank_balance_q"),
            "idle_funds": sum3("idle_funds_q"), "bank_status": latest("bank_status_q"),
        }
    else:
        cols = {
            "oss": clf_full["oss"].apply(lambda d: d[q]),
            "otr": clf_full["otr"].apply(lambda d: d[q]),
            "upto": clf_full["upto"].apply(lambda d: d[q]),
            "disb_pct": clf_full["disb_pct"].apply(lambda d: d[q]),
            "disb_amount": clf_full["disb_amount"].apply(lambda d: d[q]),
            "par": clf_full["par"].apply(lambda d: d[q]),
            "cash_balance": clf_full["cash_balance_q"].apply(lambda d: d[q]),
            "bank_balance": clf_full["bank_balance_q"].apply(lambda d: d[q]),
            "idle_funds": clf_full["idle_funds_q"].apply(lambda d: d[q]),
            "bank_status": clf_full["bank_status_q"].apply(lambda d: d[q]),
        }
    flat = pd.DataFrame({
        "mis_id": clf_full["mis_id"], "clf_name": clf_full["clf_name"],
        "district": clf_full["district"], "block": clf_full["block"], "block_id": clf_full["block_id"],
        **cols,
    })
    for ind, fn in BUCKET_FN.items():
        flat[f"{ind}_bucket"] = flat[ind].apply(fn)
    flat["bank_balance_bucket"] = flat["bank_status"]  # same 3-tier value, just named consistently with the other _bucket columns

    for ind in COMPOSITE_INDICATORS + ["par"]:
        lb = ind in LOWER_BETTER
        flat[f"{ind}_state_pctl"] = pctl(flat[ind], lower_is_better=lb)
        flat[f"{ind}_state_rank"] = flat[ind].rank(ascending=lb, method="min").astype(int)
        flat[f"{ind}_state_n"] = len(flat)

    flat["overall_score"] = flat[[f"{i}_state_pctl" for i in COMPOSITE_INDICATORS]].mean(axis=1).round(1)
    flat["overall_state_rank"] = flat["overall_score"].rank(ascending=False, method="min").astype(int)
    flat["overall_state_n"] = len(flat)

    add_group_rank(flat, "block", "block")
    add_group_rank(flat, "district", "district")
    return flat


FLAT_BY_Q = {q: build_flat_for_quarter(q) for q in PERIOD_KEYS}
print("Computed CLF-level percentiles + block/district/state ranks, per period (cumulative + 3 quarters).")

# ---------------------------------------------------------------- real per-period GROUP series
# The full monthly+quarterly mean-of-child-CLFs series, period by period -
# needed for the ALWAYS-VISIBLE "Explore by Place and Period" trend chart
# (deliberately not affected by the quarter toggle, same precedent as the
# DAK Tracker's own trend charts), and reused below to derive each
# group's OWN quarter-scoped bank-balance status from its own mean
# monthly bank-balance series.
QUARTERLY_INDS = ["oss", "otr", "upto", "disb_pct", "disb_amount", "par"]
MONTHLY_INDS = ["cash_balance", "bank_balance", "idle_funds"]

def period_matrix(ind, periods):
    df = pd.DataFrame({p: clf_full[ind].apply(lambda d: d[p]) for p in periods})
    df["district"] = clf_full["district"].values
    df["block"] = clf_full["block"].values
    return df

def group_series_lookup(group_cols):
    out = {}
    for ind in QUARTERLY_INDS:
        pm = period_matrix(ind, QUARTERS)
        out[ind] = pm.groupby(group_cols)[QUARTERS].mean().round(1) if group_cols else pm[QUARTERS].mean().round(1)
    for ind in MONTHLY_INDS:
        pm = period_matrix(ind, MONTHS)
        out[ind] = pm.groupby(group_cols)[MONTHS].mean().round(1) if group_cols else pm[MONTHS].mean().round(1)
    return out

ALL_SERIES_INDS = QUARTERLY_INDS + MONTHLY_INDS
block_series_lookup = group_series_lookup(["district", "block"])
district_series_lookup = group_series_lookup(["district"])
state_series_lookup = group_series_lookup(None)


def bucket_dist_pct(bucket_series, two_group):
    n = len(bucket_series)
    counts = bucket_series.value_counts()
    labels = ["green", "red"] if two_group else ["green", "yellow", "red"]
    return {lbl: round(float(counts.get(lbl, 0)) / n * 100, 1) if n else 0.0 for lbl in labels}


def agg_unit_for_quarter(sub, level_name, name, district=None):
    row = {"level": level_name, "name": name}
    if district: row["district"] = district
    for ind in COMPOSITE_INDICATORS + ["par", "disb_amount"]:
        row[ind] = round(float(sub[ind].mean()), 1)
    row["n_clfs"] = len(sub)
    for ind in BUCKET_INDICATORS:
        row[f"{ind}_bucket_dist"] = bucket_dist_pct(sub[f"{ind}_bucket"], ind in BUCKET_LABELS_2GROUP)
    row["cash_balance_bucket"] = cash_bucket(row["cash_balance"])
    row["idle_funds_bucket"] = idle_funds_bucket(row["idle_funds"])
    return row


def build_group_dfs_for_quarter(q):
    flat = FLAT_BY_Q[q]
    # bank_status_for_quarter needs a REAL quarter to slice 3 months out
    # of - "cumulative"'s own bank_balance is already defined as latest
    # quarter's value (see PERIOD_KEYS comment above), so its status is
    # that same latest quarter's status, not something re-derived here.
    status_q = LATEST_Q if q == "cumulative" else q

    block_rows = []
    for (d, b), sub in flat.groupby(["district", "block"]):
        r = agg_unit_for_quarter(sub, "block", b, district=d)
        r["block_id"] = sub["block_id"].iloc[0] if pd.notna(sub["block_id"].iloc[0]) else None
        r["bank_balance_bucket"] = bank_status_for_quarter(block_series_lookup["bank_balance"].loc[(d, b)].to_dict(), status_q)
        block_rows.append(r)
    block_df = pd.DataFrame(block_rows)

    district_rows = []
    for d, sub in flat.groupby("district"):
        r = agg_unit_for_quarter(sub, "district", d)
        r["district_geo_id"] = DISTRICT_ALIAS.get(str(d).strip().upper(), str(d).strip().upper())
        r["bank_balance_bucket"] = bank_status_for_quarter(district_series_lookup["bank_balance"].loc[d].to_dict(), status_q)
        district_rows.append(r)
    district_df = pd.DataFrame(district_rows)

    for ind in COMPOSITE_INDICATORS + ["par"]:
        lb = ind in LOWER_BETTER
        block_df[f"{ind}_state_pctl"] = pctl(block_df[ind], lower_is_better=lb)
        block_df[f"{ind}_state_rank"] = block_df[ind].rank(ascending=lb, method="min").astype(int)
        block_df[f"{ind}_state_n"] = len(block_df)
        district_df[f"{ind}_state_pctl"] = pctl(district_df[ind], lower_is_better=lb)
        district_df[f"{ind}_state_rank"] = district_df[ind].rank(ascending=lb, method="min").astype(int)
        district_df[f"{ind}_state_n"] = len(district_df)
    block_df["overall_score"] = block_df[[f"{i}_state_pctl" for i in COMPOSITE_INDICATORS]].mean(axis=1).round(1)
    district_df["overall_score"] = district_df[[f"{i}_state_pctl" for i in COMPOSITE_INDICATORS]].mean(axis=1).round(1)
    block_df["overall_state_rank"] = block_df["overall_score"].rank(ascending=False, method="min").astype(int)
    district_df["overall_state_rank"] = district_df["overall_score"].rank(ascending=False, method="min").astype(int)
    add_group_rank(block_df, "district", "district")

    state_row = agg_unit_for_quarter(flat, "state", "Bihar")
    state_row["bank_balance_bucket"] = bank_status_for_quarter(state_series_lookup["bank_balance"].to_dict(), status_q)

    return block_df, district_df, state_row


BLOCK_BY_Q, DISTRICT_BY_Q, STATE_BY_Q = {}, {}, {}
for q in PERIOD_KEYS:
    b_df, d_df, s_row = build_group_dfs_for_quarter(q)
    BLOCK_BY_Q[q], DISTRICT_BY_Q[q], STATE_BY_Q[q] = b_df, d_df, s_row

print(f"Aggregated {len(BLOCK_BY_Q[LATEST_Q])} blocks, {len(DISTRICT_BY_Q[LATEST_Q])} districts, 1 state, per period.")

# Attach the full (all-9-months / all-3-quarters) series to each group,
# once - unaffected by the period toggle (see docstring).
for q in PERIOD_KEYS:
    BLOCK_BY_Q[q]["series"] = BLOCK_BY_Q[q].apply(
        lambda r: {ind: block_series_lookup[ind].loc[(r["district"], r["name"])].to_dict() for ind in ALL_SERIES_INDS}, axis=1)
    DISTRICT_BY_Q[q]["series"] = DISTRICT_BY_Q[q].apply(
        lambda r: {ind: district_series_lookup[ind].loc[r["name"]].to_dict() for ind in ALL_SERIES_INDS}, axis=1)
    STATE_BY_Q[q]["series"] = {ind: state_series_lookup[ind].to_dict() for ind in ALL_SERIES_INDS}

# ---------------------------------------------------------------- write outputs
def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return None if (isinstance(obj, float) and math.isnan(obj)) else round(float(obj), 2)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.bool_,)): return bool(obj)
    return obj


def slugify(s):
    import re
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


(OUT / "clfs").mkdir(parents=True, exist_ok=True)
(OUT / "blocks").mkdir(parents=True, exist_ok=True)
(OUT / "districts").mkdir(parents=True, exist_ok=True)

# ---- CLF files: identity + by_period (one row per quarter) + the raw
# all-period series (unaffected by the toggle) ----
for _, series_row in clf_full.iterrows():
    mis_id = series_row["mis_id"]
    identity = {"mis_id": int(mis_id), "clf_name": series_row["clf_name"],
                "district": series_row["district"], "block": series_row["block"],
                "block_id": series_row["block_id"] if pd.notna(series_row["block_id"]) else None}
    by_period = {}
    for q in PERIOD_KEYS:
        r = FLAT_BY_Q[q].set_index("mis_id").loc[mis_id].to_dict()
        by_period[q] = clean(r)
    series = {k: series_row[k] for k in
              ["oss", "otr", "upto", "disb_pct", "disb_amount", "par", "cash_balance", "bank_balance", "idle_funds"]}
    payload = {**identity, "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
               "quarters": QUARTERS, "months": MONTHS, "quarter_months": QUARTER_MONTHS,
               "latest_quarter": LATEST_Q, "by_period": clean(by_period), "series": clean(series)}
    (OUT / "clfs" / f"{mis_id}.json").write_text(json.dumps(payload))

# ---- Block files ----
_all_block_names = set()
for q in QUARTERS:
    _all_block_names |= set(BLOCK_BY_Q[q].apply(lambda r: (r["district"], r["name"]), axis=1))
for district, name in sorted(_all_block_names):
    by_period = {}
    for q in PERIOD_KEYS:
        bdf = BLOCK_BY_Q[q]
        row = bdf[(bdf["district"] == district) & (bdf["name"] == name)].iloc[0]
        sub_clfs = FLAT_BY_Q[q][(FLAT_BY_Q[q]["district"] == district) & (FLAT_BY_Q[q]["block"] == name)]
        entry = clean(row.drop(labels=["series"]).to_dict())
        entry["clfs"] = clean(sub_clfs.sort_values("overall_score", ascending=False).to_dict("records"))
        by_period[q] = entry
    payload = {"level": "block", "name": name, "district": district,
               "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
               "quarters": QUARTERS, "months": MONTHS, "quarter_months": QUARTER_MONTHS, "latest_quarter": LATEST_Q,
               "by_period": by_period, "series": clean(BLOCK_BY_Q[LATEST_Q][
                   (BLOCK_BY_Q[LATEST_Q]["district"] == district) & (BLOCK_BY_Q[LATEST_Q]["name"] == name)].iloc[0]["series"])}
    (OUT / "blocks" / f"{slugify(district)}__{slugify(name)}.json").write_text(json.dumps(payload))

# ---- District files ----
for district_name in sorted(clf_df["district"].unique()):
    by_period = {}
    for q in PERIOD_KEYS:
        ddf = DISTRICT_BY_Q[q]
        row = ddf[ddf["name"] == district_name].iloc[0]
        sub_blocks = BLOCK_BY_Q[q][BLOCK_BY_Q[q]["district"] == district_name]
        sub_clfs = FLAT_BY_Q[q][FLAT_BY_Q[q]["district"] == district_name]
        entry = clean(row.drop(labels=["series"]).to_dict())
        entry["blocks"] = clean(sub_blocks.drop(columns=["series"]).sort_values("overall_score", ascending=False).to_dict("records"))
        entry["clfs"] = clean(sub_clfs.sort_values("overall_score", ascending=False).to_dict("records"))
        by_period[q] = entry
    payload = {"level": "district", "name": district_name,
               "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
               "quarters": QUARTERS, "months": MONTHS, "quarter_months": QUARTER_MONTHS, "latest_quarter": LATEST_Q,
               "by_period": by_period, "series": clean(DISTRICT_BY_Q[LATEST_Q][
                   DISTRICT_BY_Q[LATEST_Q]["name"] == district_name].iloc[0]["series"])}
    (OUT / "districts" / f"{slugify(district_name)}.json").write_text(json.dumps(payload))

# ---- State file ----
state_by_period = {}
for q in PERIOD_KEYS:
    row = dict(STATE_BY_Q[q])
    row_series = row.pop("series")
    entry = clean(row)
    entry["districts"] = clean(DISTRICT_BY_Q[q].drop(columns=["series"]).sort_values("overall_score", ascending=False).to_dict("records"))
    state_by_period[q] = entry
state_payload = {"level": "state", "name": "Bihar",
                  "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
                  "quarters": QUARTERS, "months": MONTHS, "quarter_months": QUARTER_MONTHS, "latest_quarter": LATEST_Q,
                  "by_period": state_by_period, "series": clean(STATE_BY_Q[LATEST_Q]["series"])}
(OUT / "state.json").write_text(json.dumps(state_payload))

# manifest for dropdowns (name + id hierarchy) - unaffected by quarter, unchanged
manifest_out = {"districts": []}
for d in sorted(clf_df["district"].unique()):
    blocks_out = []
    for b in sorted(clf_df.loc[clf_df["district"] == d, "block"].unique()):
        clfs_out = FLAT_BY_Q[LATEST_Q][(FLAT_BY_Q[LATEST_Q]["district"] == d) & (FLAT_BY_Q[LATEST_Q]["block"] == b)][["mis_id", "clf_name"]].to_dict("records")
        # Must exactly match the block file-naming scheme above
        # (f"{slugify(district)}__{slugify(block)}.json") - a single
        # combined slugify(f"{d}__{b}") collapses the double underscore
        # and silently produces a different, non-matching string.
        blocks_out.append({"name": b, "slug": f"{slugify(d)}__{slugify(b)}", "clfs": clfs_out})
    manifest_out["districts"].append({"name": d, "slug": slugify(d), "blocks": blocks_out})
(OUT / "manifest.json").write_text(json.dumps(clean(manifest_out)))

# compact by-quarter statewide summary for the live weight-recompute
# feature (mirrors the existing tracker's data/scoring_summary.json
# pattern, now keyed by quarter since percentiles differ per quarter).
summary = {
    "indicators": COMPOSITE_INDICATORS,
    "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
    "quarters": QUARTERS, "latest_quarter": LATEST_Q,
    "by_period": {
        q: {
            "clfs": [{"id": int(r["mis_id"]), "district": r["district"], "block": r["block"],
                      "pctl": {i: r[f"{i}_state_pctl"] for i in COMPOSITE_INDICATORS}}
                     for _, r in FLAT_BY_Q[q].iterrows()],
            "blocks": [{"name": r["name"], "district": r["district"],
                        "pctl": {i: r[f"{i}_state_pctl"] for i in COMPOSITE_INDICATORS}}
                       for _, r in BLOCK_BY_Q[q].iterrows()],
            "districts": [{"name": r["name"], "pctl": {i: r[f"{i}_state_pctl"] for i in COMPOSITE_INDICATORS}}
                          for _, r in DISTRICT_BY_Q[q].iterrows()],
        } for q in PERIOD_KEYS
    },
}
(OUT / "scoring_summary.json").write_text(json.dumps(clean(summary)))

print("\nWrote:")
print(f"  {len(list((OUT/'clfs').glob('*.json')))} CLF files")
print(f"  {len(list((OUT/'blocks').glob('*.json')))} block files")
print(f"  {len(list((OUT/'districts').glob('*.json')))} district files")
print("  state.json, manifest.json, scoring_summary.json")
print(f"\nBlock geo-match: {_matched}/{_total_b}")
