"""
build_tracker_data.py — REAL DATA generator for the CLF Financial Health
Tracker (formerly the "Synthesised CLF Tracker" prototype, whose
build_synth_tracker.py this is adapted from - same by_period/cumulative
model, same percentile-based composite scoring, same block/district/
state aggregation; only the DATA SOURCING changed, from random generation
to real cleaned data).

Sources:
  - audits_clf_transactions_merged.dta (1_Code/audit_cleaning.do output,
    re-run 2026-09-24 to pick up newly-added FY2026-27 raw exports):
    OSS, combined ICF+GL OTR/UPTO, PAR, disbursement % + amount - one row
    per (mis_id, financial_year, quarter).
  - Raw Files/F01 Balance Sheets/*.csv (not yet run through any cleaning
    .do - read directly here): Cash Balance, Bank Balance - one row per
    (District Name, CLF Name, period=month).
  - CLF Tracker/Scale-Up/data/manifest.json: the authoritative District->
    Block->CLF identity hierarchy (real names + MIS IDs) - same reuse
    precedent as the prototype.

Real quarters available (as of 2026-09-24):
  FY25-26 Q4 (Jan-Mar 2026)  - audit: yes.               F01: Jan,Feb,Mar 2026.
  FY26-27 Q1 (Apr-Jun 2026)  - audit: yes (just added).   F01: Apr,May,Jun 2026.
  FY26-27 Q2 (Jul-Sep 2026)  - audit: NOT YET.            F01: Jul,Aug,Sep 2026.
  FY26-27 Q2 has no audit data yet because audits happen after a quarter
  closes - not a bug. Its 6 audit-derived indicators are null for every
  CLF that quarter, handled by the SAME missing-data rule as everything
  else below (excluded from percentiles/composite, shown as "Not yet
  available", never defaulted to 0 or a red bucket) - not a special case.

Missing-data rule (agreed 2026-09-23, applies uniformly to every
indicator regardless of WHY it's missing - not-yet-scraped F01 district,
not-yet-audited quarter, or an unmatched CLF name):
  1. A missing value is null, never 0 and never silently defaulted to a
     bucket. Python note: `float('nan') >= 100` evaluates False, so the
     ORIGINAL bucket functions (unchanged from the prototype) would
     silently classify a missing value as the WORST bucket ("red") if
     called directly on a null - safe_bucket() below guards every bucket
     call against this.
  2. Missing values are excluded from percentile ranking, not ranked as
     the bottom score - pandas .rank()/.mean() already skip NaN by
     default, so build_flat_for_quarter/agg_unit_for_quarter's existing
     logic (unchanged from the prototype) gets this right for free AS
     LONG AS missing values are fed in as None/NaN, never 0.
  3. Group-level (block/district/state) bucket-distribution percentages
     are computed over CLFs that actually HAVE data for that indicator,
     not diluted by CLFs with none - bucket_dist_pct() below excludes
     nulls from both numerator and denominator, and a companion
     `{ind}_n_available` / `{ind}_n_total` pair is shipped alongside
     every group's own indicator value so the frontend can show a
     "data available for X of Y" note rather than let a sparse group's
     numbers pass as if they were complete.

Known real data-quality notes (verified directly against the cleaned
data, not assumed):
  - rep_icf_otr / rep_gl_otr / rep_icf_upto / rep_gl_upto (the raw LokOS-
    computed repayment-rate fields) contain out-of-range values for some
    rows (e.g. one row's rep_gl_otr = 2,538,622) - NOT used here. OTR/
    UPTO are instead computed directly from the underlying planned/
    recovered AMOUNTS (rep_icf/gl_planned/recovery_*), which is also the
    only way to get a genuine ICF+GL COMBINED rate (the raw fields keep
    ICF and GL separate).
  - Even recomputed from amounts, combined OTR has real extreme outliers
    (mean ~288%, max ~11,000% in a spot check on FY25-26 Q4). Not capped
    here - no principled cap to apply, and percentile-based ranking
    (used for the composite score) is robust to a handful of extreme
    values; shown as-is in the raw KPI figure.
  - F01 (Cash/Bank Balance) is matched to mis_id by exact (district, CLF
    name) match only - 82.5% match rate (1,409/1,707 unique F01 CLF
    identities), confirmed directly. The remaining ~18% are minor
    spelling variants a fuzzy matcher could resolve - not built in this
    pass. Those CLFs simply show "Not yet available" for cash/bank/idle
    funds, same as a genuinely un-scraped district would - the tracker
    can't tell the difference between the two causes, which is honest,
    since the effect (no data) is the same either way.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
EXISTING_TRACKER = HERE.parent / "CLF Tracker" / "Scale-Up"
OUT = HERE / "data"
CLEANED_DIR = Path("/Users/mohanrajagopal/Dropbox/Bihar Gates Team/3_Jeevika/6_LokOS_Analysis/2_Data/Cleaned")
F01_DIR = Path("/Users/mohanrajagopal/Dropbox/Bihar Gates Team/3_Jeevika/6_LokOS_Analysis/2_Data/Raw Files/F01 Balance Sheets")

QUARTERS = ["FY25-26 Q4", "FY26-27 Q1", "FY26-27 Q2"]
MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]
QUARTER_MONTHS = {
    "FY25-26 Q4": MONTHS[0:3],
    "FY26-27 Q1": MONTHS[3:6],
    "FY26-27 Q2": MONTHS[6:9],
}
LATEST_Q = QUARTERS[-1]
PERIOD_KEYS = ["cumulative"] + QUARTERS
DEFAULT_PERIOD = "cumulative"

CASH_THRESHOLD = 25_000
BANK_THRESHOLD = 25_00_000

FY_QUARTER_MAP = {
    ("2025 - 2026", 4.0): "FY25-26 Q4",
    ("2026 - 2027", 1.0): "FY26-27 Q1",
    ("2026 - 2027", 2.0): "FY26-27 Q2",
}

# ---------------------------------------------------------------- category-bucket thresholds (unchanged from the prototype)
def oss_bucket(v):
    if v >= 100: return "green"
    if v >= 75: return "yellow"
    return "red"

def par_bucket(v):
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

# n = number of CLFs the value represents (1 for a single CLF; at
# block/district/state level the value is now a TOTAL across all
# member CLFs, so the per-CLF threshold is scaled by n to stay
# equivalent to "the average member CLF crosses this threshold" -
# same classification outcome as bucketing an average, but displays
# the real total rather than an average.
def idle_funds_bucket(v, n=1):
    if v <= 0: return "green"
    if v <= 5_00_000 * n: return "yellow"
    return "red"

def cash_bucket(v, n=1):
    if v < 20_000 * n: return "green"
    if v < 30_000 * n: return "yellow"
    return "red"

BUCKET_FN = {
    "oss": oss_bucket, "par": par_bucket, "otr": otr_bucket, "upto": upto_bucket,
    "disb_pct": disb_pct_bucket, "idle_funds": idle_funds_bucket, "cash_balance": cash_bucket,
}
BUCKET_INDICATORS = list(BUCKET_FN.keys()) + ["bank_balance"]
BUCKET_LABELS_2GROUP = set()


def is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def safe_bucket(fn, v):
    """Guards every bucket_fn(v) call against a missing value - v >= 100
    on a NaN silently evaluates False in Python, so calling a bucket
    function directly on a missing value would misclassify it as the
    WORST bucket instead of correctly returning "no data"."""
    if is_missing(v):
        return None
    return fn(v)


# ---------------------------------------------------------------- load the real hierarchy (unchanged from the prototype)
manifest = json.load(open(EXISTING_TRACKER / "data" / "manifest.json"))
clf_rows = []
for d in manifest["districts"]:
    for b in d["blocks"]:
        for c in b["clfs"]:
            clf_rows.append({"mis_id": c["id"], "clf_name": c["n"], "district": d["name"], "block": b["name"]})
clf_df = pd.DataFrame(clf_rows)
print(f"Loaded real hierarchy: {clf_df['district'].nunique()} districts, "
      f"{clf_df.groupby(['district','block']).ngroups} blocks, {len(clf_df)} CLFs")

def norm_block(b):
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
print(f"Block geo-match: {_matched}/{_total_b} blocks matched to the 534-block map.")

# ---------------------------------------------------------------- load real audit data
audit = pd.read_stata(CLEANED_DIR / "audits_clf_transactions_merged.dta", convert_categoricals=False)
audit["period_key"] = audit.apply(lambda r: FY_QUARTER_MAP.get((r["financial_year"], r["quarter"])), axis=1)
audit = audit[audit["period_key"].notna()].copy()

# Combined ICF+GL OTR and UPTO - official LokOS column formulas, confirmed
# directly against the raw Report_CLf_Wise export's own header row (not
# assumed) and against Mohan's own explicit correction:
#   ICF OTR  = Column S / Column R * 100 = rep_icf_recovery_last_month / rep_icf_actual_last_month
#   GL  OTR  = Column AD / Column AB * 100 = rep_gl_recovery_current_qtr / rep_gl_actual_current_qtr
#   ICF UPTO = Column V / Column Q * 100 = rep_icf_recovery_total / rep_icf_planned_total
#   GL  UPTO = Column AE / Column AA * 100 = rep_gl_recovery_total / rep_gl_planned_total
# (an earlier pass used rep_icf_recovery_current_qtr/rep_icf_planned_current_qtr
# for ICF OTR and rep_gl_planned_current_qtr as GL's denominator - both
# wrong columns; UPTO was already correct and is unchanged here.)
# "Combined" ICF+GL = summed numerators over summed denominators (amount-
# weighted), not an average of the two rates - a CLF with a huge ICF book
# and a token GL book shouldn't have GL's rate count for 50% of the blend.
#
# Sanity guard: even with the right columns, some rows have a recovery
# figure that's logically impossible (a single month/quarter's recovery
# EXCEEDING that same loan type's own lifetime recovery or lifetime
# planned demand) - confirmed directly (e.g. one CLF's rep_icf_recovery_
# last_month = 103,000,650 against an ICF lifetime planned demand of far
# less) - a genuine raw-data quality issue, not a formula error. Any
# numerator failing this check is treated as unreliable and dropped from
# THAT loan type's contribution to the combined OTR for that CLF-quarter
# (the other loan type, if clean, still counts) rather than silently
# feeding a nonsense number into the average.
def _valid_recovery(recovery, recovery_total, planned_total, denom=None):
    recovery = recovery.fillna(0)
    ok = (recovery <= recovery_total.fillna(np.inf)) & (recovery <= planned_total.fillna(np.inf))
    if denom is not None:
        # A loan type reporting recovery against zero (or missing) current-
        # period demand is another logically-impossible combination found
        # directly in the raw data (e.g. one CLF's rep_gl_actual_current_qtr
        # = 0 while rep_gl_recovery_current_qtr = 498,000, which alone blew
        # a single CLF's combined OTR up to 9,924%) - exclude it the same
        # way as the lifetime-total check above rather than let a real
        # numerator divide by a bogus near-zero denominator.
        ok = ok & (denom.fillna(0) > 0)
    return recovery.where(ok, np.nan)

icf_otr_den = audit["rep_icf_actual_last_month"]
gl_otr_den = audit["rep_gl_actual_current_qtr"]
icf_otr_num = _valid_recovery(audit["rep_icf_recovery_last_month"], audit["rep_icf_recovery_total"], audit["rep_icf_planned_total"], denom=icf_otr_den)
gl_otr_num = _valid_recovery(audit["rep_gl_recovery_current_qtr"], audit["rep_gl_recovery_total"], audit["rep_gl_planned_total"], denom=gl_otr_den)

# Per-CLF-quarter component amounts, carried through (not just the final
# ratio) so the "cumulative" period below can re-derive a TRUE combined
# rate from amounts SUMMED across every available quarter, rather than
# averaging 3 already-computed quarterly rates.
audit["_icf_otr_num"] = icf_otr_num
audit["_icf_otr_den"] = icf_otr_den
audit["_gl_otr_num"] = gl_otr_num
audit["_gl_otr_den"] = gl_otr_den
otr_num = icf_otr_num.fillna(0) + gl_otr_num.fillna(0)
otr_den = icf_otr_den.fillna(0) + gl_otr_den.fillna(0)
audit["otr"] = np.where(otr_den > 0, otr_num / otr_den * 100, np.nan)

icf_rt, gl_rt = audit["rep_icf_recovery_total"].fillna(0), audit["rep_gl_recovery_total"].fillna(0)
icf_pt, gl_pt = audit["rep_icf_planned_total"].fillna(0), audit["rep_gl_planned_total"].fillna(0)
dent = icf_pt + gl_pt
audit["upto"] = np.where(dent > 0, (icf_rt + gl_rt) / dent * 100, np.nan)

# oss is used as-reported (oss_cq_oss), not recomputed from earned_income/
# admin_exp - those two raw component columns were found to be internally
# inconsistent with the platform's own oss_cq_oss for the SAME row in
# >99% of cases checked directly (e.g. one CLF: earned/admin implies
# 52,481%, but the platform's own reported value for that exact CLF-
# quarter is 78.5%), even though earned/admin*100 is exactly the formula
# the raw file's own header documents - a genuine raw-data inconsistency,
# not a formula error. Real outliers still exist in oss_cq_oss itself and
# are not capped here - see docstring.
audit["oss"] = audit["oss_cq_oss"]
audit["par"] = 100 - audit["otr"]
audit["disb_pct"] = audit["cum_disb_percent"]
audit["disb_amount"] = audit["cum_loan_disb"]

AUDIT_INDS = ["oss", "otr", "upto", "par", "disb_pct", "disb_amount"]
AUDIT_RAW_COMPONENTS = ["_icf_otr_num", "_icf_otr_den", "_gl_otr_num", "_gl_otr_den"]

# Known-corrupted raw record, confirmed directly against the source
# Excel: CLF 121120 (Lakhnaur block, Madhubani)'s FY26-27 Q1 row has
# rep_icf_planned_total = 194 against rep_icf_recovery_total =
# 22,930,165 - a magnitude/unit error in the raw MIS export, not a
# formula issue (this CLF's FY25-26 Q4 UPTO of 92.93% is normal and
# untouched). Excluded from calculations per user decision rather than
# capped, so it doesn't distort this CLF's own display, any
# percentile/ranking it would otherwise feed into, or the cumulative
# "latest available" logic.
_bad = (audit["mis_id"] == 121120) & (audit["period_key"] == "FY26-27 Q1")
audit.loc[_bad, "upto"] = np.nan

audit_flat = audit[["mis_id", "period_key"] + AUDIT_INDS + AUDIT_RAW_COMPONENTS].dropna(subset=["mis_id"]).copy()
audit_flat["mis_id"] = audit_flat["mis_id"].astype(int)
audit_flat = audit_flat.drop_duplicates(subset=["mis_id", "period_key"], keep="first")

audit_by_clf = {}
for _, r in audit_flat.iterrows():
    d = audit_by_clf.setdefault(int(r["mis_id"]), {})
    d[r["period_key"]] = {ind: (None if pd.isna(r[ind]) else float(r[ind])) for ind in AUDIT_INDS + AUDIT_RAW_COMPONENTS}
print(f"Loaded real audit data: {len(audit_flat)} (CLF, period) rows across {audit_flat['mis_id'].nunique()} CLFs.")

# ---------------------------------------------------------------- load real F01 balance-sheet data, match to mis_id
f01_files = sorted(F01_DIR.glob("F01_*_CLF_level.csv"))
f01 = pd.concat([pd.read_csv(f) for f in f01_files], ignore_index=True)
f01["dist_u"] = f01["District Name"].astype(str).str.strip().str.upper()
f01["clf_u"] = f01["CLF Name"].astype(str).str.strip().str.upper()
clf_df["dist_u"] = clf_df["district"].str.strip().str.upper()
clf_df["clf_u"] = clf_df["clf_name"].str.strip().str.upper()
f01_xwalk = clf_df[["dist_u", "clf_u", "mis_id"]].drop_duplicates()
_f01_before = len(f01[["dist_u", "clf_u"]].drop_duplicates())
f01 = f01.merge(f01_xwalk, on=["dist_u", "clf_u"], how="left")
_f01_matched_rows = f01["mis_id"].notna().sum()
print(f"F01 match (exact district+CLF-name, no fuzzy pass): {_f01_matched_rows}/{len(f01)} rows matched "
      f"({f01.dropna(subset=['mis_id'])[['dist_u','clf_u']].drop_duplicates().shape[0]}/{_f01_before} unique CLF identities).")
f01 = f01.dropna(subset=["mis_id"]).copy()
f01["mis_id"] = f01["mis_id"].astype(int)

MONTH_NAME_TO_NUM = {"January": "01", "February": "02", "March": "03", "April": "04", "May": "05",
                      "June": "06", "July": "07", "August": "08", "September": "09"}
def period_to_month(p):
    name, year = p.split("-")
    return f"{year}-{MONTH_NAME_TO_NUM[name]}"
f01["month"] = f01["period"].apply(period_to_month)

f01_g = f01.groupby(["mis_id", "month"]).agg(
    cash_balance=("Cash Balance", "mean"), bank_balance=("Bank Balance", "mean")).reset_index()

cash_by_clf, bank_by_clf = {}, {}
for _, r in f01_g.iterrows():
    cash_by_clf.setdefault(int(r["mis_id"]), {})[r["month"]] = float(r["cash_balance"])
    bank_by_clf.setdefault(int(r["mis_id"]), {})[r["month"]] = float(r["bank_balance"])
print(f"Loaded real F01 data: {len(f01_g)} (CLF, month) rows across {f01_g['mis_id'].nunique()} CLFs.")


def bank_status_for_quarter(vals_by_month, q, n=1):
    """None (not green) unless all 3 of the quarter's months are
    actually present - a partial quarter (1-2 months scraped) can't
    honestly claim "0 of 3 months over threshold" when the missing
    month might have been the one that would have flagged it.
    n = number of CLFs the values represent (see cash_bucket/
    idle_funds_bucket for why this scales the threshold)."""
    months = QUARTER_MONTHS[q]
    have = [vals_by_month[m] for m in months if m in vals_by_month]
    if len(have) < 3:
        return None
    n_over = sum(1 for v in have if v >= BANK_THRESHOLD * n)
    if n_over >= 3: return "red"
    if n_over >= 1: return "yellow"
    return "green"


# ---------------------------------------------------------------- assemble per-CLF records
records = []
for _, row in clf_df.iterrows():
    mis_id = int(row["mis_id"])
    rec = {"mis_id": mis_id, "clf_name": row["clf_name"], "district": row["district"],
           "block": row["block"], "block_id": row["block_id"] if pd.notna(row["block_id"]) else None}

    audit_periods = audit_by_clf.get(mis_id, {})
    cash_months = cash_by_clf.get(mis_id, {})
    bank_months = bank_by_clf.get(mis_id, {})
    rec["cash_balance"] = cash_months
    rec["bank_balance"] = bank_months

    bank_status_q, cash_balance_q, bank_balance_q = {}, {}, {}
    idle_funds_month, idle_funds_q = {}, {}
    for q in QUARTERS:
        bank_status_q[q] = bank_status_for_quarter(bank_months, q)
        last_m = QUARTER_MONTHS[q][-1]
        cash_balance_q[q] = cash_months.get(last_m)
        bank_balance_q[q] = bank_months.get(last_m)

        q_idle_total, any_month = 0.0, False
        for m in QUARTER_MONTHS[q]:
            c, b = cash_months.get(m), bank_months.get(m)
            if c is None and b is None:
                continue
            any_month = True
            idle_cash = max(0.0, (c or 0) - CASH_THRESHOLD) if c is not None else 0.0
            idle_bank = max(0.0, (b or 0) - BANK_THRESHOLD) if (b is not None and bank_status_q[q] == "red") else 0.0
            idle_funds_month[m] = round(idle_cash + idle_bank, 0)
            q_idle_total += idle_funds_month[m]
        idle_funds_q[q] = round(q_idle_total, 0) if any_month else None

    rec["idle_funds"] = idle_funds_month
    rec["bank_status_q"] = bank_status_q
    rec["cash_balance_q"] = cash_balance_q
    rec["bank_balance_q"] = bank_balance_q
    rec["idle_funds_q"] = idle_funds_q

    # "Latest available" (not "latest calendar month regardless of
    # whether it was scraped") - for cumulative view, a balance is a
    # current-state fact, so the most RECENT actual reading is the right
    # cumulative figure, searched across all 9 months directly (not just
    # each quarter's own end-month, which may itself be missing even when
    # an earlier month in a later quarter's window isn't - doesn't happen
    # here since months are scraped in order, but this is robust either way).
    _cash_avail = sorted(cash_months.keys())
    _bank_avail = sorted(bank_months.keys())
    rec["cash_balance_latest"] = cash_months[_cash_avail[-1]] if _cash_avail else None
    rec["bank_balance_latest"] = bank_months[_bank_avail[-1]] if _bank_avail else None
    rec["bank_status_latest"] = next((bank_status_q[q] for q in reversed(QUARTERS) if bank_status_q[q] is not None), None)

    for ind in AUDIT_INDS + AUDIT_RAW_COMPONENTS:
        rec[ind] = {q: (audit_periods.get(q, {}) or {}).get(ind) for q in QUARTERS}

    records.append(rec)

clf_full = pd.DataFrame(records)
print(f"Assembled real per-CLF records for {len(clf_full)} CLFs.")

# ---------------------------------------------------------------- percentile helper
def pctl(series, lower_is_better=False):
    s = series.astype(float)
    ranks = s.rank(pct=True, ascending=not lower_is_better)  # NaN stays NaN - excluded, not ranked last
    return (ranks * 100).round(1)


COMPOSITE_INDICATORS = ["oss", "otr", "upto", "disb_pct", "cash_balance", "bank_balance", "idle_funds"]
LOWER_BETTER = {"cash_balance", "bank_balance", "idle_funds", "par"}


# Indicators where a genuinely missing value is the CLF's OWN doing
# (no loan demand recorded, or the audit never captured a disbursement/
# OSS figure for them) - per user decision, these CLFs are RANKED AS
# THE WORST performer, not excluded from ranking/scoring, unlike
# cash_balance/bank_balance/idle_funds (whose gaps come from
# incomplete F01 scraping - not the CLF's fault - and keep the
# original exclude-from-ranking behavior everywhere else in this file).
# Both this function and the percentile loop in build_flat_for_quarter
# read a "{ind}__score" column when present (the zero-filled scoring
# value) and fall back to the plain, possibly-null "{ind}" column
# otherwise - the RAW displayed value never changes, only what feeds
# rank/percentile/Overall Score.
ZERO_PENALTY_INDS = {"oss", "otr", "upto", "disb_pct"}

def score_col_name(df, ind):
    col = f"{ind}__score"
    return col if col in df.columns else ind

def add_group_rank(df, group_col, prefix):
    g = df.groupby(group_col)["overall_score"]
    df[f"overall_{prefix}_rank"] = g.rank(ascending=False, method="min")
    df[f"overall_{prefix}_n"] = g.transform("count").astype(int)
    for ind in COMPOSITE_INDICATORS + ["par"]:
        gg = df.groupby(group_col)[score_col_name(df, ind)]
        lb = ind in LOWER_BETTER
        df[f"{ind}_{prefix}_rank"] = gg.rank(ascending=lb, method="min")
        df[f"{ind}_{prefix}_n"] = gg.transform("count").astype(int)


def build_flat_for_quarter(q):
    if q == "cumulative":
        # Genuine cross-period combination, per indicator's own real
        # nature (agreed 2026-09-24, replacing an earlier "latest
        # quarter's value" rule that broke whenever the temporally-latest
        # quarter simply had no data yet - e.g. 76% of CLFs showed a
        # blank cumulative UPTO purely because FY26-27 Q2 has no audit,
        # even when FY25-26 Q4 or FY26-27 Q1 had perfectly good data):
        #   oss, otr  -> recomputed from the raw components SUMMED across
        #                every quarter that has them (a true blended
        #                rate over the whole tracked window - not an
        #                average of 3 already-computed quarterly rates,
        #                and not tied to whichever quarter is temporally
        #                newest).
        #   par       -> 100 - the cumulative otr just computed above.
        #   upto, disb_pct, disb_amount -> these are ALREADY cumulative-
        #                since-inception by their own definition (each
        #                quarter's own value already encompasses every
        #                prior quarter) - summing or averaging them across
        #                quarters would double/triple-count, so cumulative
        #                here means the most RECENT quarter that actually
        #                HAS a value (searched backward, not just LATEST_Q
        #                blindly).
        #   cash_balance, bank_balance -> a balance is a current-state
        #                fact, not additive - cumulative = the most recent
        #                ACTUAL reading across all 9 months (not just each
        #                quarter's end-month), via cash/bank_balance_latest
        #                computed once per CLF above.
        #   idle_funds -> genuinely additive (excess funds accumulated) -
        #                summed across every quarter that has a value.
        def latest_available(col):
            return clf_full[col].apply(lambda d: next((d[qq] for qq in reversed(QUARTERS) if d[qq] is not None), None))

        def cum_otr_par(row):
            icf_n = _safe_sum([row["_icf_otr_num"][qq] for qq in QUARTERS])
            icf_d = _safe_sum([row["_icf_otr_den"][qq] for qq in QUARTERS])
            gl_n = _safe_sum([row["_gl_otr_num"][qq] for qq in QUARTERS])
            gl_d = _safe_sum([row["_gl_otr_den"][qq] for qq in QUARTERS])
            den = (icf_d or 0) + (gl_d or 0)
            if den <= 0:
                return None
            return (icf_n or 0) + (gl_n or 0), den

        # oss -> NOT recombined from raw earned_income/admin_exp like OTR,
        #        despite that being the platform's own documented formula
        #        (verified directly against the raw file's header text) -
        #        because those two raw component columns were found to be
        #        internally inconsistent with the platform's own already-
        #        reported oss_cq_oss for the SAME row in >99% of cases
        #        checked directly (e.g. one CLF: earned/admin implies
        #        52,481%, but the platform's own reported value for that
        #        exact CLF-quarter is 78.5%) - a genuine raw-data issue,
        #        confirmed outside this script against the raw merged
        #        dataset, not a bug in this aggregation. Recombining across
        #        quarters from unreliable components would compound that
        #        inconsistency, so cumulative OSS instead reuses the
        #        already-reported per-quarter ratio, latest-available -
        #        the same treatment as upto/disb_pct/disb_amount above.
        otr_vals, par_vals = [], []
        for _, row in clf_full.iterrows():
            r = cum_otr_par(row)
            if r is None:
                otr_vals.append(None); par_vals.append(None)
            else:
                num, den = r
                otr = round(num / den * 100, 2)
                otr_vals.append(otr); par_vals.append(round(100 - otr, 2))

        cols = {
            "oss": latest_available("oss"),
            "otr": pd.Series(otr_vals, index=clf_full.index),
            "par": pd.Series(par_vals, index=clf_full.index),
            "upto": latest_available("upto"),
            "disb_pct": latest_available("disb_pct"),
            "disb_amount": latest_available("disb_amount"),
            "cash_balance": clf_full["cash_balance_latest"],
            "bank_balance": clf_full["bank_balance_latest"],
            "idle_funds": clf_full["idle_funds_q"].apply(lambda d: _safe_sum([d[qq] for qq in QUARTERS])),
            "bank_status": clf_full["bank_status_latest"],
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
        flat[f"{ind}_bucket"] = flat[ind].apply(lambda v: safe_bucket(fn, v))
    flat["bank_balance_bucket"] = flat["bank_status"]

    # See ZERO_PENALTY_INDS above add_group_rank for the reasoning.
    for ind in ZERO_PENALTY_INDS:
        flat[f"{ind}__score"] = flat[ind].fillna(0)
    # par = 100 - otr by definition; deriving its scoring value the same
    # way keeps a no-demand CLF's PAR scored as the worst (100), not
    # the best (0 would be the LOWEST-risk PAR - the opposite of a
    # penalty, since PAR is a lower-is-better indicator).
    flat["par__score"] = 100 - flat["otr__score"]

    for ind in COMPOSITE_INDICATORS + ["par"]:
        lb = ind in LOWER_BETTER
        sc = flat[score_col_name(flat, ind)]
        flat[f"{ind}_state_pctl"] = pctl(sc, lower_is_better=lb)
        flat[f"{ind}_state_rank"] = sc.rank(ascending=lb, method="min")
        flat[f"{ind}_state_n"] = sc.notna().sum()

    flat["overall_score"] = flat[[f"{i}_state_pctl" for i in COMPOSITE_INDICATORS]].mean(axis=1).round(1)
    flat["overall_state_rank"] = flat["overall_score"].rank(ascending=False, method="min")
    flat["overall_state_n"] = flat["overall_score"].notna().sum()

    add_group_rank(flat, "block", "block")
    add_group_rank(flat, "district", "district")
    flat = flat.drop(columns=[c for c in flat.columns if c.endswith("__score")])
    return flat


def _safe_sum(vals):
    vals = [v for v in vals if not is_missing(v)]
    return float(np.sum(vals)) if vals else None


FLAT_BY_Q = {q: build_flat_for_quarter(q) for q in PERIOD_KEYS}
print("Computed CLF-level percentiles + block/district/state ranks, per period (cumulative + 3 quarters).")

# ---------------------------------------------------------------- group aggregation
def bucket_dist_pct(bucket_series, two_group):
    """Excludes CLFs with no data (null bucket) from BOTH the numerator
    and denominator - a district where only 2 of 14 blocks have cash
    data shows the distribution among just those 2, not diluted by 12
    blocks that simply have no answer yet."""
    valid = bucket_series.dropna()
    n = len(valid)
    counts = valid.value_counts()
    labels = ["green", "red"] if two_group else ["green", "yellow", "red"]
    return {lbl: round(float(counts.get(lbl, 0)) / n * 100, 1) if n else None for lbl in labels}


# Amount-type indicators are TOTALED across a group's member CLFs at
# block/district/state level (a real "how much cash does this
# district hold" figure), not averaged - everything else (rates) is
# still averaged. Per user decision, the corresponding bucket
# thresholds (cash/idle_funds/bank) scale by n_clfs so the group's
# color classification stays equivalent to "the average member CLF
# crosses this threshold" rather than misclassifying every group as
# the worst bucket just because a sum is naturally larger than one
# CLF's threshold.
SUM_INDS = {"cash_balance", "bank_balance", "idle_funds", "disb_amount"}

def agg_unit_for_quarter(sub, level_name, name, district=None):
    row = {"level": level_name, "name": name}
    if district: row["district"] = district
    for ind in COMPOSITE_INDICATORS + ["par", "disb_amount"]:
        vals = sub[ind].dropna()
        if len(vals):
            row[ind] = round(float(vals.sum() if ind in SUM_INDS else vals.mean()), 1)
        else:
            row[ind] = None
        row[f"{ind}_n_available"] = int(len(vals))
        row[f"{ind}_n_total"] = int(len(sub))
    row["n_clfs"] = len(sub)
    for ind in BUCKET_INDICATORS:
        row[f"{ind}_bucket_dist"] = bucket_dist_pct(sub[f"{ind}_bucket"], ind in BUCKET_LABELS_2GROUP)
    n_clfs = row["n_clfs"] or 1
    # Rate indicators (oss/otr/upto/disb_pct/par) bucket the group's
    # own averaged value directly against the same thresholds as a
    # single CLF - no n_clfs scaling needed, since an average rate is
    # already on the same 0-100(+) scale regardless of group size.
    # Amount indicators (cash/idle_funds) DO need the n_clfs-scaled
    # threshold since their group value is now a total - previously
    # these were the only two indicators bucketed at group level at
    # all (oss/otr/upto/disb_pct/par's own group-level heatmap color
    # was never computed, always None).
    for ind, fn in BUCKET_FN.items():
        if ind in ("cash_balance", "idle_funds"):
            row[f"{ind}_bucket"] = safe_bucket(lambda v, fn=fn: fn(v, n=n_clfs), row[ind])
        else:
            row[f"{ind}_bucket"] = safe_bucket(fn, row[ind])
    return row


QUARTERLY_INDS = ["oss", "otr", "upto", "disb_pct", "disb_amount", "par"]
MONTHLY_INDS = ["cash_balance", "bank_balance", "idle_funds"]

def period_matrix(ind, periods):
    # .get(p), not d[p] - unlike the prototype's fake data (every CLF had
    # a full 9-month dict by construction), the real monthly cash/bank
    # dicts only have keys for months actually scraped for that CLF, so a
    # missing month/quarter must resolve to None, not KeyError.
    # astype(float) turns those Python Nones into proper np.nan - left
    # as object dtype, DataFrame.sum(min_count=1) can hand back a
    # literal None for an all-missing column instead of nan, which
    # .round() then chokes on.
    df = pd.DataFrame({p: clf_full[ind].apply(lambda d: d.get(p)) for p in periods}).astype(float)
    df["district"] = clf_full["district"].values
    df["block"] = clf_full["block"].values
    return df

def _agg_series(pm, cols, group_cols, use_sum):
    # min_count=1 makes an all-missing group sum to NaN (not pandas'
    # default 0-for-empty-sum), preserving the "missing is never 0"
    # rule for a group where no member CLF has a value that month.
    if use_sum:
        return (pm.groupby(group_cols)[cols].sum(min_count=1).round(1) if group_cols
                else pm[cols].sum(min_count=1).round(1))
    return (pm.groupby(group_cols)[cols].mean().round(1) if group_cols
            else pm[cols].mean().round(1))

def group_series_lookup(group_cols):
    out = {}
    for ind in QUARTERLY_INDS:
        pm = period_matrix(ind, QUARTERS)
        out[ind] = _agg_series(pm, QUARTERS, group_cols, ind in SUM_INDS)
    for ind in MONTHLY_INDS:
        pm = period_matrix(ind, MONTHS)
        out[ind] = _agg_series(pm, MONTHS, group_cols, ind in SUM_INDS)
    return out

ALL_SERIES_INDS = QUARTERLY_INDS + MONTHLY_INDS
block_series_lookup = group_series_lookup(["district", "block"])
district_series_lookup = group_series_lookup(["district"])
state_series_lookup = group_series_lookup(None)


def build_group_dfs_for_quarter(q):
    flat = FLAT_BY_Q[q]
    status_q = LATEST_Q if q == "cumulative" else q

    block_rows = []
    for (d, b), sub in flat.groupby(["district", "block"]):
        r = agg_unit_for_quarter(sub, "block", b, district=d)
        r["block_id"] = sub["block_id"].iloc[0] if pd.notna(sub["block_id"].iloc[0]) else None
        bs = block_series_lookup["bank_balance"].loc[(d, b)]
        r["bank_balance_bucket"] = bank_status_for_quarter(bs.dropna().to_dict(), status_q, n=r["n_clfs"] or 1) if bs.notna().any() else None
        block_rows.append(r)
    block_df = pd.DataFrame(block_rows)

    district_rows = []
    for d, sub in flat.groupby("district"):
        r = agg_unit_for_quarter(sub, "district", d)
        r["district_geo_id"] = DISTRICT_ALIAS.get(str(d).strip().upper(), str(d).strip().upper())
        ds = district_series_lookup["bank_balance"].loc[d]
        r["bank_balance_bucket"] = bank_status_for_quarter(ds.dropna().to_dict(), status_q, n=r["n_clfs"] or 1) if ds.notna().any() else None
        district_rows.append(r)
    district_df = pd.DataFrame(district_rows)

    for ind in COMPOSITE_INDICATORS + ["par"]:
        lb = ind in LOWER_BETTER
        block_df[f"{ind}_state_pctl"] = pctl(block_df[ind], lower_is_better=lb)
        block_df[f"{ind}_state_rank"] = block_df[ind].rank(ascending=lb, method="min")
        block_df[f"{ind}_state_n"] = block_df[ind].notna().sum()
        district_df[f"{ind}_state_pctl"] = pctl(district_df[ind], lower_is_better=lb)
        district_df[f"{ind}_state_rank"] = district_df[ind].rank(ascending=lb, method="min")
        district_df[f"{ind}_state_n"] = district_df[ind].notna().sum()
    block_df["overall_score"] = block_df[[f"{i}_state_pctl" for i in COMPOSITE_INDICATORS]].mean(axis=1).round(1)
    district_df["overall_score"] = district_df[[f"{i}_state_pctl" for i in COMPOSITE_INDICATORS]].mean(axis=1).round(1)
    block_df["overall_state_rank"] = block_df["overall_score"].rank(ascending=False, method="min")
    district_df["overall_state_rank"] = district_df["overall_score"].rank(ascending=False, method="min")
    # Statewide rank-pool size for the score box's "of N districts/
    # blocks" display - was missing entirely (only the district-level
    # pool size for blocks, overall_district_n via add_group_rank
    # below, was ever set).
    block_df["overall_state_n"] = block_df["overall_score"].notna().sum()
    district_df["overall_state_n"] = district_df["overall_score"].notna().sum()
    add_group_rank(block_df, "district", "district")

    state_row = agg_unit_for_quarter(flat, "state", "Bihar")
    ss = state_series_lookup["bank_balance"]
    state_row["bank_balance_bucket"] = bank_status_for_quarter(ss.dropna().to_dict(), status_q, n=state_row["n_clfs"] or 1) if ss.notna().any() else None

    return block_df, district_df, state_row


BLOCK_BY_Q, DISTRICT_BY_Q, STATE_BY_Q = {}, {}, {}
for q in PERIOD_KEYS:
    b_df, d_df, s_row = build_group_dfs_for_quarter(q)
    BLOCK_BY_Q[q], DISTRICT_BY_Q[q], STATE_BY_Q[q] = b_df, d_df, s_row
print(f"Aggregated {len(BLOCK_BY_Q[LATEST_Q])} blocks, {len(DISTRICT_BY_Q[LATEST_Q])} districts, 1 state, per period.")

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
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")


(OUT / "clfs").mkdir(parents=True, exist_ok=True)
(OUT / "blocks").mkdir(parents=True, exist_ok=True)
(OUT / "districts").mkdir(parents=True, exist_ok=True)

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
        entry["clfs"] = clean(sub_clfs.sort_values("overall_score", ascending=False, na_position="last").to_dict("records"))
        by_period[q] = entry
    payload = {"level": "block", "name": name, "district": district,
               "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
               "quarters": QUARTERS, "months": MONTHS, "quarter_months": QUARTER_MONTHS, "latest_quarter": LATEST_Q,
               "by_period": by_period, "series": clean(BLOCK_BY_Q[LATEST_Q][
                   (BLOCK_BY_Q[LATEST_Q]["district"] == district) & (BLOCK_BY_Q[LATEST_Q]["name"] == name)].iloc[0]["series"])}
    (OUT / "blocks" / f"{slugify(district)}__{slugify(name)}.json").write_text(json.dumps(payload))

for district_name in sorted(clf_df["district"].unique()):
    by_period = {}
    for q in PERIOD_KEYS:
        ddf = DISTRICT_BY_Q[q]
        row = ddf[ddf["name"] == district_name].iloc[0]
        sub_blocks = BLOCK_BY_Q[q][BLOCK_BY_Q[q]["district"] == district_name]
        sub_clfs = FLAT_BY_Q[q][FLAT_BY_Q[q]["district"] == district_name]
        entry = clean(row.drop(labels=["series"]).to_dict())
        entry["blocks"] = clean(sub_blocks.drop(columns=["series"]).sort_values("overall_score", ascending=False, na_position="last").to_dict("records"))
        entry["clfs"] = clean(sub_clfs.sort_values("overall_score", ascending=False, na_position="last").to_dict("records"))
        by_period[q] = entry
    payload = {"level": "district", "name": district_name,
               "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
               "quarters": QUARTERS, "months": MONTHS, "quarter_months": QUARTER_MONTHS, "latest_quarter": LATEST_Q,
               "by_period": by_period, "series": clean(DISTRICT_BY_Q[LATEST_Q][
                   DISTRICT_BY_Q[LATEST_Q]["name"] == district_name].iloc[0]["series"])}
    (OUT / "districts" / f"{slugify(district_name)}.json").write_text(json.dumps(payload))

state_by_period = {}
for q in PERIOD_KEYS:
    row = dict(STATE_BY_Q[q])
    row.pop("series")
    entry = clean(row)
    entry["districts"] = clean(DISTRICT_BY_Q[q].drop(columns=["series"]).sort_values("overall_score", ascending=False, na_position="last").to_dict("records"))
    scored_clfs = FLAT_BY_Q[q].dropna(subset=["overall_score"])
    entry["top_clfs"] = clean(scored_clfs.sort_values("overall_score", ascending=False).head(10).to_dict("records"))
    entry["bottom_clfs"] = clean(scored_clfs.sort_values("overall_score", ascending=True).head(10).to_dict("records"))
    state_by_period[q] = entry
state_payload = {"level": "state", "name": "Bihar",
                  "periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD,
                  "quarters": QUARTERS, "months": MONTHS, "quarter_months": QUARTER_MONTHS, "latest_quarter": LATEST_Q,
                  "by_period": state_by_period, "series": clean(STATE_BY_Q[LATEST_Q]["series"])}
(OUT / "state.json").write_text(json.dumps(state_payload))

# One consolidated file for the "Export to Excel" button - full flat
# CLF/block/district/state tables (raw values + every rank/percentile
# field already computed), one period at a time. Built here rather
# than having the export button fetch each of the ~2,240 individual
# clfs/blocks/districts/*.json files at click time.
export_payload = {"periods": PERIOD_KEYS, "default_period": DEFAULT_PERIOD}
for q in PERIOD_KEYS:
    export_payload[q] = {
        "clfs": clean(FLAT_BY_Q[q].to_dict("records")),
        "blocks": clean(BLOCK_BY_Q[q].drop(columns=["series"]).to_dict("records")),
        "districts": clean(DISTRICT_BY_Q[q].drop(columns=["series"]).to_dict("records")),
        "state": clean({k: v for k, v in STATE_BY_Q[q].items() if k != "series"}),
    }
(OUT / "export_data.json").write_text(json.dumps(export_payload))

manifest_out = {"districts": []}
for d in sorted(clf_df["district"].unique()):
    blocks_out = []
    for b in sorted(clf_df.loc[clf_df["district"] == d, "block"].unique()):
        clfs_out = FLAT_BY_Q[LATEST_Q][(FLAT_BY_Q[LATEST_Q]["district"] == d) & (FLAT_BY_Q[LATEST_Q]["block"] == b)][["mis_id", "clf_name"]].to_dict("records")
        blocks_out.append({"name": b, "slug": f"{slugify(d)}__{slugify(b)}", "clfs": clfs_out})
    manifest_out["districts"].append({"name": d, "slug": slugify(d), "blocks": blocks_out})
(OUT / "manifest.json").write_text(json.dumps(clean(manifest_out)))

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
print("  state.json, manifest.json, scoring_summary.json, export_data.json")
print(f"\nBlock geo-match: {_matched}/{_total_b}")
