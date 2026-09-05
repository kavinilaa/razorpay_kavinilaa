import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TXN_PARQUET = os.path.join(ROOT, "data", "processed", "engineered_transactions.parquet")
TIME_PARQUET = os.path.join(ROOT, "data", "processed", "time_window_features.parquet")
FIG_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

print("Reading columns needed for figures...")
cols = ["isFraud", "type", "log_amount", "drained_to_zero", "hist_dest_txn_count",
        "high_amount_indicator", "amount_to_dest_avg_ratio"]
df = pd.read_parquet(TXN_PARQUET, columns=cols)
print(f"Loaded {len(df):,} rows, columns: {cols}")

# 08: log_amount distribution fraud vs legit
fig, ax = plt.subplots(figsize=(7, 4))
legit = df.loc[df["isFraud"] == 0, "log_amount"]
fraud = df.loc[df["isFraud"] == 1, "log_amount"]
ax.hist(legit, bins=80, alpha=0.6, label="Legit", color="#4C72B0", density=True)
ax.hist(fraud, bins=80, alpha=0.6, label="Fraud", color="#C44E52", density=True)
ax.set_xlabel("log_amount"); ax.set_ylabel("Density")
ax.set_title("Engineered log_amount: Fraud vs Legit")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "08_log_amount_by_class.png"))
plt.close(fig)

# 09: drained_to_zero rate by class
fig, ax = plt.subplots(figsize=(5, 4))
rates = df.groupby("isFraud")["drained_to_zero"].mean() * 100
ax.bar(["Legit", "Fraud"], rates.values, color=["#4C72B0", "#C44E52"])
ax.set_ylabel("% of transactions with drained_to_zero == 1")
ax.set_title("Origin-Drained-to-Zero Rate: Fraud vs Legit")
for i, v in enumerate(rates.values):
    ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "09_drained_to_zero_by_class.png"))
plt.close(fig)

# 10: hist_dest_txn_count distribution (log scale) by class
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(np.log1p(df.loc[df["isFraud"] == 0, "hist_dest_txn_count"]), bins=40, alpha=0.6,
        label="Legit", color="#4C72B0", density=True)
ax.hist(np.log1p(df.loc[df["isFraud"] == 1, "hist_dest_txn_count"]), bins=40, alpha=0.6,
        label="Fraud", color="#C44E52", density=True)
ax.set_xlabel("log1p(hist_dest_txn_count)"); ax.set_ylabel("Density")
ax.set_title("Destination Prior-History Count (leakage-safe): Fraud vs Legit")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "10_hist_dest_txn_count_by_class.png"))
plt.close(fig)

# 11: high_amount_indicator rate by type and class
fig, ax = plt.subplots(figsize=(7, 4))
pivot = df.groupby(["type", "isFraud"], observed=True)["high_amount_indicator"].mean().unstack() * 100
pivot.plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
ax.set_ylabel("% high_amount_indicator == 1")
ax.set_title("High-Amount Flag Rate by Transaction Type and Class")
ax.legend(["Legit", "Fraud"])
ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "11_high_amount_by_type_class.png"))
plt.close(fig)

del df

# 12: spike-detection ratio over time
print("Reading time_window_features...")
twf = pd.read_parquet(TIME_PARQUET)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(twf["step"], twf["txn_count_vs_hour_baseline_ratio"], linewidth=0.8, color="#4C72B0")
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="baseline (ratio=1)")
ax.set_xlabel("Step (hour)"); ax.set_ylabel("txn_count / hour-of-day historical baseline")
ax.set_title("Volume-Conditioned Spike Signal Over Time")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "12_volume_baseline_ratio_over_time.png"))
plt.close(fig)

# 13: gt_fraud_rate vs hist baseline (evaluation-only, for illustration)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(twf["step"], twf["gt_fraud_rate"] * 100, linewidth=0.8, color="#DD8452", label="actual fraud rate (%)")
ax.plot(twf["step"], twf["gt_hist_avg_fraud_rate_same_hour"] * 100, linewidth=0.8, color="black",
        linestyle="--", label="historical baseline for this hour (%)")
ax.set_xlabel("Step (hour)"); ax.set_ylabel("Fraud rate (%)")
ax.set_title("Ground-Truth Fraud Rate vs. Hour-of-Day Baseline (evaluation only, not a model input)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "13_fraud_rate_vs_baseline_eval_only.png"))
plt.close(fig)

print("Figures written to", FIG_DIR)

stats = {
    "drained_to_zero_rate_by_class_pct": {str(k): float(v) for k, v in rates.items()},
}
with open(os.path.join(ROOT, "reports", "phase2_figure_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("Done.")
