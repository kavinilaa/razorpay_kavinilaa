"""
Phase 1 - Dataset Analysis for AI Fraud-Spike & Risk Detection System
Loads the PaySim dataset, computes all statistics needed for the Phase 1
report, saves figures to reports/figures/, and dumps every computed number
to reports/phase1_stats.json so the markdown report can quote real values.

Read-only with respect to the source CSV: data/raw/PS_20174392719_1491204439457_log.csv
is never modified or rewritten.
"""
import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 140)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "data", "raw", "PS_20174392719_1491204439457_log.csv")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
STATS_PATH = os.path.join(ROOT, "reports", "phase1_stats.json")
os.makedirs(FIG_DIR, exist_ok=True)

stats = {}

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# ---------------------------------------------------------------------------
# 1. Load dataset with memory-efficient dtypes
# ---------------------------------------------------------------------------
log("Loading dataset...")
t0 = time.time()

dtypes = {
    "step": "int32",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "int8",
    "isFlaggedFraud": "int8",
}

df = pd.read_csv(CSV_PATH, dtype=dtypes)
load_time = time.time() - t0
log(f"Loaded {len(df):,} rows in {load_time:.1f}s")

n_rows, n_cols = df.shape
stats["shape"] = {"rows": int(n_rows), "cols": int(n_cols)}
stats["columns"] = list(df.columns)
stats["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
mem_bytes = df.memory_usage(deep=True).sum()
stats["memory_usage_mb"] = round(mem_bytes / (1024 ** 2), 2)
stats["load_time_sec"] = round(load_time, 2)
stats["source_file_size_mb"] = round(os.path.getsize(CSV_PATH) / (1024 ** 2), 2)

log(f"Rows={n_rows:,} Cols={n_cols} Memory={stats['memory_usage_mb']} MB")

# ---------------------------------------------------------------------------
# 2. Data quality: missing values, duplicates
# ---------------------------------------------------------------------------
log("Checking data quality...")
missing = df.isna().sum()
stats["missing_values"] = {c: int(v) for c, v in missing.items()}
stats["total_missing"] = int(missing.sum())

dup_count = int(df.duplicated().sum())
stats["duplicate_rows"] = dup_count

# categorical unique values
stats["type_unique_values"] = df["type"].value_counts().to_dict()
stats["type_unique_values"] = {str(k): int(v) for k, v in stats["type_unique_values"].items()}

n_unique_orig = df["nameOrig"].nunique()
n_unique_dest = df["nameDest"].nunique()
stats["unique_nameOrig"] = int(n_unique_orig)
stats["unique_nameDest"] = int(n_unique_dest)
stats["n_steps"] = int(df["step"].nunique())
stats["step_min"] = int(df["step"].min())
stats["step_max"] = int(df["step"].max())

# origin/destination account prefix check (C = customer, M = merchant)
orig_prefixes = df["nameOrig"].str[0].value_counts().to_dict()
dest_prefixes = df["nameDest"].str[0].value_counts().to_dict()
stats["nameOrig_prefixes"] = {str(k): int(v) for k, v in orig_prefixes.items()}
stats["nameDest_prefixes"] = {str(k): int(v) for k, v in dest_prefixes.items()}

# accounts appearing as both orig and dest
origs = set(df["nameOrig"].unique())
dests = set(df["nameDest"].unique())
overlap = origs & dests
stats["accounts_both_orig_and_dest"] = len(overlap)
del origs, dests, overlap

log(f"Missing total={stats['total_missing']} Duplicates={dup_count} "
    f"UniqueOrig={n_unique_orig:,} UniqueDest={n_unique_dest:,} Steps={stats['n_steps']}")

# ---------------------------------------------------------------------------
# 3. Target analysis: isFraud
# ---------------------------------------------------------------------------
log("Analyzing target variable...")
fraud_counts = df["isFraud"].value_counts().to_dict()
n_fraud = int(fraud_counts.get(1, 0))
n_legit = int(fraud_counts.get(0, 0))
fraud_pct = n_fraud / n_rows * 100
stats["isFraud_counts"] = {"0": n_legit, "1": n_fraud}
stats["fraud_percentage"] = round(fraud_pct, 6)
stats["imbalance_ratio_legit_to_fraud"] = round(n_legit / n_fraud, 1) if n_fraud else None

log(f"Fraud={n_fraud:,} ({fraud_pct:.4f}%) Legit={n_legit:,} "
    f"Ratio={stats['imbalance_ratio_legit_to_fraud']}:1")

# ---------------------------------------------------------------------------
# 4. isFlaggedFraud analysis
# ---------------------------------------------------------------------------
flagged_counts = df["isFlaggedFraud"].value_counts().to_dict()
n_flagged = int(flagged_counts.get(1, 0))
stats["isFlaggedFraud_counts"] = {"0": int(flagged_counts.get(0, 0)), "1": n_flagged}

flagged_and_fraud = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 1)).sum())
flagged_not_fraud = int(((df["isFlaggedFraud"] == 1) & (df["isFraud"] == 0)).sum())
fraud_not_flagged = int(((df["isFlaggedFraud"] == 0) & (df["isFraud"] == 1)).sum())
stats["isFlaggedFraud_analysis"] = {
    "total_flagged": n_flagged,
    "flagged_and_actually_fraud": flagged_and_fraud,
    "flagged_but_not_fraud": flagged_not_fraud,
    "fraud_but_not_flagged": fraud_not_flagged,
    "recall_of_flag_on_fraud": round(flagged_and_fraud / n_fraud, 6) if n_fraud else None,
}

if n_flagged:
    flagged_rows = df[df["isFlaggedFraud"] == 1]
    stats["isFlaggedFraud_analysis"]["flagged_types"] = {
        str(k): int(v) for k, v in flagged_rows["type"].value_counts().to_dict().items()
    }
    stats["isFlaggedFraud_analysis"]["flagged_amount_min"] = float(flagged_rows["amount"].min())
    stats["isFlaggedFraud_analysis"]["flagged_amount_max"] = float(flagged_rows["amount"].max())

log(f"isFlaggedFraud total={n_flagged}, of which actually fraud={flagged_and_fraud}, "
    f"fraud missed by flag={fraud_not_flagged}")

# ---------------------------------------------------------------------------
# 5. Fraud by transaction type
# ---------------------------------------------------------------------------
log("Analyzing fraud by transaction type...")
by_type = df.groupby("type", observed=True).agg(
    total_txn=("isFraud", "size"),
    fraud_txn=("isFraud", "sum"),
).reset_index()
by_type["fraud_rate_pct"] = by_type["fraud_txn"] / by_type["total_txn"] * 100
stats["fraud_by_type"] = by_type.set_index("type").to_dict(orient="index")
stats["fraud_by_type"] = {str(k): {kk: (int(vv) if kk != "fraud_rate_pct" else round(float(vv), 6))
                                    for kk, vv in v.items()} for k, v in stats["fraud_by_type"].items()}

fraud_only_types = sorted(df.loc[df["isFraud"] == 1, "type"].unique().tolist())
stats["fraud_only_transaction_types"] = [str(t) for t in fraud_only_types]

log(f"Fraud occurs only in types: {fraud_only_types}")

# ---------------------------------------------------------------------------
# 6. Amount column statistics
# ---------------------------------------------------------------------------
log("Analyzing amount distribution...")
amt = df["amount"]
percentiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]
amt_pct = amt.quantile(percentiles)
stats["amount_stats"] = {
    "min": float(amt.min()),
    "max": float(amt.max()),
    "mean": float(amt.mean()),
    "median": float(amt.median()),
    "std": float(amt.std()),
    "percentiles": {f"p{int(p*100) if p*100==int(p*100) else p*100}": float(v) for p, v in amt_pct.items()},
    "zero_amount_count": int((amt == 0).sum()),
}

fraud_amt = df.loc[df["isFraud"] == 1, "amount"]
legit_amt = df.loc[df["isFraud"] == 0, "amount"]
stats["amount_stats_by_class"] = {
    "fraud": {"mean": float(fraud_amt.mean()), "median": float(fraud_amt.median()),
              "min": float(fraud_amt.min()), "max": float(fraud_amt.max()), "std": float(fraud_amt.std())},
    "legit": {"mean": float(legit_amt.mean()), "median": float(legit_amt.median()),
              "min": float(legit_amt.min()), "max": float(legit_amt.max()), "std": float(legit_amt.std())},
}

# amount range buckets and fraud rate per bucket
bins = [0, 1000, 10000, 50000, 100000, 200000, 500000, 1000000, np.inf]
labels = ["0-1K", "1K-10K", "10K-50K", "50K-100K", "100K-200K", "200K-500K", "500K-1M", "1M+"]
df["amount_bucket"] = pd.cut(df["amount"], bins=bins, labels=labels, right=False)
by_bucket = df.groupby("amount_bucket", observed=True).agg(
    total_txn=("isFraud", "size"), fraud_txn=("isFraud", "sum")
)
by_bucket["fraud_rate_pct"] = by_bucket["fraud_txn"] / by_bucket["total_txn"] * 100
stats["fraud_by_amount_bucket"] = {
    str(k): {"total_txn": int(v["total_txn"]), "fraud_txn": int(v["fraud_txn"]),
              "fraud_rate_pct": round(float(v["fraud_rate_pct"]), 6)}
    for k, v in by_bucket.to_dict(orient="index").items()
}

log("Amount stats computed.")

# ---------------------------------------------------------------------------
# 7. Balance columns - suspicious pattern detection
# ---------------------------------------------------------------------------
log("Analyzing balance columns...")

# Expected: newbalanceOrig = oldbalanceOrg - amount (for a normal debit)
orig_balance_error = (df["oldbalanceOrg"] - df["amount"]) - df["newbalanceOrig"]
stats["orig_balance_exact_match_pct"] = round(float((orig_balance_error.abs() < 0.01).mean() * 100), 4)

# Expected: newbalanceDest = oldbalanceDest + amount
dest_balance_error = (df["oldbalanceDest"] + df["amount"]) - df["newbalanceDest"]
stats["dest_balance_exact_match_pct"] = round(float((dest_balance_error.abs() < 0.01).mean() * 100), 4)

# zero-balance patterns often flagged in PaySim literature as fraud signals
orig_zero_pattern = ((df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0) & (df["amount"] > 0))
dest_zero_pattern = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0) & (df["amount"] > 0))

stats["balance_patterns"] = {
    "orig_old_and_new_zero_but_amount_gt0": {
        "count": int(orig_zero_pattern.sum()),
        "fraud_rate_pct": round(float(df.loc[orig_zero_pattern, "isFraud"].mean() * 100), 4) if orig_zero_pattern.any() else 0.0,
    },
    "dest_old_and_new_zero_but_amount_gt0": {
        "count": int(dest_zero_pattern.sum()),
        "fraud_rate_pct": round(float(df.loc[dest_zero_pattern, "isFraud"].mean() * 100), 4) if dest_zero_pattern.any() else 0.0,
    },
    "overall_fraud_rate_pct": round(fraud_pct, 4),
}

# balance inconsistency vs fraud correlation (mismatch on origin side)
orig_mismatch = orig_balance_error.abs() > 0.01
stats["balance_patterns"]["orig_balance_mismatch"] = {
    "count": int(orig_mismatch.sum()),
    "fraud_rate_pct": round(float(df.loc[orig_mismatch, "isFraud"].mean() * 100), 4),
}
dest_mismatch = dest_balance_error.abs() > 0.01
stats["balance_patterns"]["dest_balance_mismatch"] = {
    "count": int(dest_mismatch.sum()),
    "fraud_rate_pct": round(float(df.loc[dest_mismatch, "isFraud"].mean() * 100), 4),
}

# zero balances overall
stats["zero_balance_counts"] = {
    "oldbalanceOrg_zero": int((df["oldbalanceOrg"] == 0).sum()),
    "newbalanceOrig_zero": int((df["newbalanceOrig"] == 0).sum()),
    "oldbalanceDest_zero": int((df["oldbalanceDest"] == 0).sum()),
    "newbalanceDest_zero": int((df["newbalanceDest"] == 0).sum()),
}

# fraud broken down: does fraud drain origin account to exactly zero?
fraud_rows = df[df["isFraud"] == 1]
drained_to_zero = (fraud_rows["newbalanceOrig"] == 0) & (fraud_rows["oldbalanceOrg"] > 0)
stats["fraud_drains_origin_to_zero_pct"] = round(float(drained_to_zero.mean() * 100), 2) if len(fraud_rows) else None

del orig_balance_error, dest_balance_error, orig_zero_pattern, dest_zero_pattern, orig_mismatch, dest_mismatch, fraud_rows, drained_to_zero

log("Balance pattern analysis complete.")

# ---------------------------------------------------------------------------
# 8. Temporal analysis (per step)
# ---------------------------------------------------------------------------
log("Analyzing temporal behavior...")
per_step = df.groupby("step").agg(
    txn_count=("isFraud", "size"),
    fraud_count=("isFraud", "sum"),
    total_amount=("amount", "sum"),
    avg_amount=("amount", "mean"),
).reset_index()
per_step["fraud_rate_pct"] = per_step["fraud_count"] / per_step["txn_count"] * 100

stats["temporal"] = {
    "txn_count_per_step_stats": {
        "min": int(per_step["txn_count"].min()), "max": int(per_step["txn_count"].max()),
        "mean": round(float(per_step["txn_count"].mean()), 2), "std": round(float(per_step["txn_count"].std()), 2),
    },
    "fraud_rate_per_step_stats": {
        "min": round(float(per_step["fraud_rate_pct"].min()), 4),
        "max": round(float(per_step["fraud_rate_pct"].max()), 4),
        "mean": round(float(per_step["fraud_rate_pct"].mean()), 4),
        "std": round(float(per_step["fraud_rate_pct"].std()), 4),
    },
    "top10_steps_by_fraud_rate": per_step.nlargest(10, "fraud_rate_pct")[
        ["step", "txn_count", "fraud_count", "fraud_rate_pct"]].to_dict(orient="records"),
    "top10_steps_by_fraud_count": per_step.nlargest(10, "fraud_count")[
        ["step", "txn_count", "fraud_count", "fraud_rate_pct"]].to_dict(orient="records"),
    "top10_steps_by_txn_volume": per_step.nlargest(10, "txn_count")[
        ["step", "txn_count", "fraud_count", "fraud_rate_pct"]].to_dict(orient="records"),
    "steps_with_zero_fraud": int((per_step["fraud_count"] == 0).sum()),
}

# step -> implied hour of day/day of week (PaySim: 1 step = 1 hour, 744 steps = 31 days)
per_step["hour_of_day"] = per_step["step"] % 24
per_step["day"] = (per_step["step"] - 1) // 24 + 1
hourly = per_step.groupby("hour_of_day").agg(
    total_txn=("txn_count", "sum"), total_fraud=("fraud_count", "sum")
)
hourly["fraud_rate_pct"] = hourly["total_fraud"] / hourly["total_txn"] * 100
stats["hourly_pattern"] = hourly.reset_index().to_dict(orient="records")
stats["hourly_pattern"] = [
    {"hour_of_day": int(r["hour_of_day"]), "total_txn": int(r["total_txn"]),
     "total_fraud": int(r["total_fraud"]), "fraud_rate_pct": round(float(r["fraud_rate_pct"]), 4)}
    for r in stats["hourly_pattern"]
]

# spike detection: flag steps where fraud_rate is > mean + 2*std, or txn_count is > mean + 2*std
fr_mean, fr_std = per_step["fraud_rate_pct"].mean(), per_step["fraud_rate_pct"].std()
vol_mean, vol_std = per_step["txn_count"].mean(), per_step["txn_count"].std()
fraud_spike_steps = per_step.loc[per_step["fraud_rate_pct"] > fr_mean + 2 * fr_std, "step"].tolist()
volume_spike_steps = per_step.loc[per_step["txn_count"] > vol_mean + 2 * vol_std, "step"].tolist()
stats["spike_detection"] = {
    "fraud_rate_threshold_mean_plus_2std": round(float(fr_mean + 2 * fr_std), 4),
    "n_steps_flagged_as_fraud_spike": len(fraud_spike_steps),
    "fraud_spike_steps_sample": fraud_spike_steps[:30],
    "volume_threshold_mean_plus_2std": round(float(vol_mean + 2 * vol_std), 2),
    "n_steps_flagged_as_volume_spike": len(volume_spike_steps),
    "volume_spike_steps_sample": volume_spike_steps[:30],
}

log(f"Fraud-rate spikes (>{fr_mean+2*fr_std:.2f}%): {len(fraud_spike_steps)} steps. "
    f"Volume spikes: {len(volume_spike_steps)} steps.")

# ---------------------------------------------------------------------------
# 9. Behavioral / velocity features feasibility
# ---------------------------------------------------------------------------
log("Analyzing behavioral feature feasibility...")
orig_txn_counts = df["nameOrig"].value_counts()
dest_txn_counts = df["nameDest"].value_counts()
stats["behavioral"] = {
    "nameOrig_appears_more_than_once": int((orig_txn_counts > 1).sum()),
    "nameOrig_max_txn_count": int(orig_txn_counts.max()),
    "nameOrig_single_txn_pct": round(float((orig_txn_counts == 1).mean() * 100), 2),
    "nameDest_appears_more_than_once": int((dest_txn_counts > 1).sum()),
    "nameDest_max_txn_count": int(dest_txn_counts.max()),
    "nameDest_repeat_pct": round(float((dest_txn_counts > 1).mean() * 100), 2),
    "top10_busiest_destinations": {str(k): int(v) for k, v in dest_txn_counts.head(10).items()},
}
del orig_txn_counts, dest_txn_counts

log("Behavioral analysis complete.")

# ---------------------------------------------------------------------------
# 10. Save stats JSON
# ---------------------------------------------------------------------------
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2, default=str)
log(f"Stats written to {STATS_PATH}")

# ---------------------------------------------------------------------------
# 11. Visualizations
# ---------------------------------------------------------------------------
log("Generating visualizations...")

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

# 1. Fraud vs non-fraud distribution (log scale bar)
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Legit (0)", "Fraud (1)"], [n_legit, n_fraud], color=["#4C72B0", "#C44E52"])
ax.set_yscale("log")
ax.set_ylabel("Transaction count (log scale)")
ax.set_title(f"Fraud vs Non-Fraud Distribution\n(fraud = {fraud_pct:.4f}% of {n_rows:,} txns)")
for i, v in enumerate([n_legit, n_fraud]):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_fraud_distribution.png"))
plt.close(fig)

# 2. Transactions over time (per step)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(per_step["step"], per_step["txn_count"], linewidth=0.8, color="#4C72B0")
ax.set_xlabel("Step (hour)")
ax.set_ylabel("Transaction count")
ax.set_title("Transaction Volume Over Time (per step)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_transactions_over_time.png"))
plt.close(fig)

# 3. Fraud count over time
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(per_step["step"], per_step["fraud_count"], linewidth=0.8, color="#C44E52")
ax.set_xlabel("Step (hour)")
ax.set_ylabel("Fraud transaction count")
ax.set_title("Fraud Count Over Time (per step)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_fraud_count_over_time.png"))
plt.close(fig)

# 4. Fraud rate over time
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(per_step["step"], per_step["fraud_rate_pct"], linewidth=0.8, color="#DD8452")
ax.axhline(fr_mean, color="gray", linestyle="--", linewidth=1, label=f"mean={fr_mean:.3f}%")
ax.axhline(fr_mean + 2 * fr_std, color="red", linestyle="--", linewidth=1, label=f"mean+2std={fr_mean+2*fr_std:.3f}%")
ax.set_xlabel("Step (hour)")
ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud Rate Over Time (per step)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_fraud_rate_over_time.png"))
plt.close(fig)

# 5. Transaction type distribution
fig, ax = plt.subplots(figsize=(6, 4))
type_counts = df["type"].value_counts()
ax.bar(type_counts.index.astype(str), type_counts.values, color="#4C72B0")
ax.set_ylabel("Transaction count")
ax.set_title("Transaction Type Distribution")
ax.tick_params(axis="x", rotation=30)
for i, v in enumerate(type_counts.values):
    ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "05_transaction_type_distribution.png"))
plt.close(fig)

# 6. Fraud by transaction type
fig, ax1 = plt.subplots(figsize=(6, 4))
bt = by_type.set_index("type")
ax1.bar(bt.index.astype(str), bt["fraud_txn"], color="#C44E52")
ax1.set_ylabel("Fraud transaction count", color="#C44E52")
ax1.tick_params(axis="x", rotation=30)
ax2 = ax1.twinx()
ax2.plot(bt.index.astype(str), bt["fraud_rate_pct"], color="black", marker="o", linewidth=1)
ax2.set_ylabel("Fraud rate (%)")
ax1.set_title("Fraud Count and Fraud Rate by Transaction Type")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "06_fraud_by_transaction_type.png"))
plt.close(fig)

# 7. Transaction amount distribution (log scale, fraud vs legit)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log1p(legit_amt), bins=80, alpha=0.6, label="Legit", color="#4C72B0", density=True)
ax.hist(np.log1p(fraud_amt), bins=80, alpha=0.6, label="Fraud", color="#C44E52", density=True)
ax.set_xlabel("log(1 + amount)")
ax.set_ylabel("Density")
ax.set_title("Transaction Amount Distribution (log scale): Fraud vs Legit")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "07_amount_distribution.png"))
plt.close(fig)

log("Visualizations saved to reports/figures/")

total_time = time.time() - t0
stats["total_analysis_time_sec"] = round(total_time, 1)
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2, default=str)

log(f"DONE. Total time: {total_time:.1f}s")
