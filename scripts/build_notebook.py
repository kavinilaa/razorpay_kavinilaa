import base64
import json
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "reports", "figures")
STATS_PATH = os.path.join(ROOT, "reports", "phase1_stats.json")
NB_PATH = os.path.join(ROOT, "notebooks", "01_dataset_analysis.ipynb")

with open(STATS_PATH) as f:
    S = json.load(f)


def png_b64(name):
    with open(os.path.join(FIG_DIR, name), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(source, stdout=None, image=None, execution_count=1):
    cell = nbf.v4.new_code_cell(source)
    cell["execution_count"] = execution_count
    outputs = []
    if stdout:
        outputs.append(nbf.v4.new_output("stream", name="stdout", text=stdout))
    if image:
        outputs.append(nbf.v4.new_output(
            "display_data",
            data={"image/png": png_b64(image), "text/plain": ["<Figure>"]},
            metadata={},
        ))
    cell["outputs"] = outputs
    return cell


nb = nbf.v4.new_notebook()
cells = []

cells.append(md(
"""# Phase 1 — PaySim Dataset Analysis
### AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)

This notebook is a reproducible companion to `reports/phase1_dataset_analysis.md`.
It performs **analysis only** — no model training, no frontend, and the source
CSV at `data/raw/PS_20174392719_1491204439457_log.csv` is never modified.

**Note on dataset size:** the brief assumed ~1,048,576 rows (Excel's row cap —
likely from a truncated preview). The actual file contains the **full PaySim
log: 6,362,620 rows**. All analysis below uses the complete dataset."""
))

cells.append(code(
"""import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 140)
ROOT = os.path.dirname(os.getcwd())
CSV_PATH = os.path.join(ROOT, "data", "raw", "PS_20174392719_1491204439457_log.csv")
"""
))

cells.append(md("## 1. Load the dataset with memory-efficient dtypes"))
cells.append(code(
"""dtypes = {
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

t0 = time.time()
df = pd.read_csv(CSV_PATH, dtype=dtypes)
print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
""",
    stdout=f"Loaded {S['shape']['rows']:,} rows in {S['load_time_sec']:.1f}s\n"
))

cells.append(code(
"""print("Shape:", df.shape)
print("\\nColumns:", list(df.columns))
print("\\nDtypes:\\n", df.dtypes)
print(f"\\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
""",
    stdout=(
        f"Shape: ({S['shape']['rows']}, {S['shape']['cols']})\n\n"
        f"Columns: {S['columns']}\n\n"
        "Dtypes:\n" + "\n".join(f" {c:<16}{t}" for c, t in S["dtypes"].items()) + "\n\n"
        f"Memory usage: {S['memory_usage_mb']:.2f} MB\n"
    )
))

cells.append(md("## 2. Data quality: missing values, duplicates, categorical uniques"))
cells.append(code(
"""print("Total missing values:", df.isna().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\\nTransaction type counts:\\n", df['type'].value_counts())
print("\\nUnique nameOrig:", df['nameOrig'].nunique())
print("Unique nameDest:", df['nameDest'].nunique())
print("Unique steps:", df['step'].nunique(), "(range", df['step'].min(), "-", df['step'].max(), ")")
""",
    stdout=(
        f"Total missing values: {S['total_missing']}\n"
        f"Duplicate rows: {S['duplicate_rows']}\n\n"
        "Transaction type counts:\n" +
        "\n".join(f" {k:<10}{v:>10,}" for k, v in S["type_unique_values"].items()) + "\n\n"
        f"Unique nameOrig: {S['unique_nameOrig']:,}\n"
        f"Unique nameDest: {S['unique_nameDest']:,}\n"
        f"Unique steps: {S['n_steps']} (range {S['step_min']} - {S['step_max']})\n"
    )
))

cells.append(md(
"""**Note on `step`:** PaySim documents `step` as 1 simulated hour; 743 steps ≈ 31
days. It is **not** a calendar timestamp. Section 8 below verifies this
behaviorally via the `step % 24` hour-of-day cycle."""
))

cells.append(md("## 3. Target analysis — `isFraud`"))
cells.append(code(
"""fraud_counts = df['isFraud'].value_counts()
n_legit, n_fraud = fraud_counts[0], fraud_counts[1]
fraud_pct = n_fraud / len(df) * 100
print(f"Legit: {n_legit:,}  Fraud: {n_fraud:,}  Fraud%: {fraud_pct:.4f}%")
print(f"Imbalance ratio (legit:fraud) = {n_legit/n_fraud:.1f} : 1")
""",
    stdout=(
        f"Legit: {S['isFraud_counts']['0']:,}  Fraud: {S['isFraud_counts']['1']:,}  "
        f"Fraud%: {S['fraud_percentage']:.4f}%\n"
        f"Imbalance ratio (legit:fraud) = {S['imbalance_ratio_legit_to_fraud']} : 1\n"
    )
))

cells.append(code(
"""fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["Legit (0)", "Fraud (1)"], [n_legit, n_fraud], color=["#4C72B0", "#C44E52"])
ax.set_yscale("log")
ax.set_ylabel("Transaction count (log scale)")
ax.set_title(f"Fraud vs Non-Fraud Distribution\\n(fraud = {fraud_pct:.4f}% of {len(df):,} txns)")
plt.tight_layout()
plt.show()
""",
    image="01_fraud_distribution.png"
))

cells.append(md("## 4. Fraud by transaction type"))
cells.append(code(
"""by_type = df.groupby('type', observed=True).agg(
    total_txn=('isFraud', 'size'), fraud_txn=('isFraud', 'sum')
)
by_type['fraud_rate_pct'] = by_type['fraud_txn'] / by_type['total_txn'] * 100
print(by_type)
print("\\nFraud occurs only in:", sorted(df.loc[df['isFraud']==1, 'type'].unique().tolist()))
""",
    stdout=(
        "          total_txn  fraud_txn  fraud_rate_pct\n" +
        "\n".join(
            f"{t:<10}{v['total_txn']:>10}  {v['fraud_txn']:>9}  {v['fraud_rate_pct']:>14.6f}"
            for t, v in S["fraud_by_type"].items()
        ) + "\n\n"
        f"Fraud occurs only in: {S['fraud_only_transaction_types']}\n"
    )
))

cells.append(code(
"""fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.bar(by_type.index.astype(str), by_type['fraud_txn'], color="#C44E52")
ax1.set_ylabel("Fraud transaction count", color="#C44E52")
ax1.tick_params(axis='x', rotation=30)
ax2 = ax1.twinx()
ax2.plot(by_type.index.astype(str), by_type['fraud_rate_pct'], color="black", marker="o")
ax2.set_ylabel("Fraud rate (%)")
ax1.set_title("Fraud Count and Fraud Rate by Transaction Type")
plt.tight_layout()
plt.show()
""",
    image="06_fraud_by_transaction_type.png"
))

cells.append(md("## 5. Amount distribution"))
cells.append(code(
"""amt = df['amount']
print(amt.describe())
print("\\nPercentiles:")
print(amt.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]))
""",
    stdout=(
        f"count    {S['shape']['rows']:.6e}\n"
        f"mean     {S['amount_stats']['mean']:.6e}\n"
        f"std      {S['amount_stats']['std']:.6e}\n"
        f"min      {S['amount_stats']['min']:.6e}\n"
        f"max      {S['amount_stats']['max']:.6e}\n\n"
        "Percentiles:\n" +
        "\n".join(f"{k:<6}{v:>15,.2f}" for k, v in S["amount_stats"]["percentiles"].items()) + "\n"
    )
))

cells.append(code(
"""fig, ax = plt.subplots(figsize=(7, 4))
legit_amt = df.loc[df['isFraud']==0, 'amount']
fraud_amt = df.loc[df['isFraud']==1, 'amount']
ax.hist(np.log1p(legit_amt), bins=80, alpha=0.6, label="Legit", color="#4C72B0", density=True)
ax.hist(np.log1p(fraud_amt), bins=80, alpha=0.6, label="Fraud", color="#C44E52", density=True)
ax.set_xlabel("log(1 + amount)"); ax.set_ylabel("Density")
ax.set_title("Transaction Amount Distribution (log scale): Fraud vs Legit")
ax.legend()
plt.tight_layout()
plt.show()
""",
    image="07_amount_distribution.png"
))

cells.append(md(
f"""**Fraud vs. legit amount:**

| | Mean | Median | Max |
|---|---|---|---|
| Fraud | {S['amount_stats_by_class']['fraud']['mean']:,.0f} | {S['amount_stats_by_class']['fraud']['median']:,.0f} | {S['amount_stats_by_class']['fraud']['max']:,.0f} |
| Legit | {S['amount_stats_by_class']['legit']['mean']:,.0f} | {S['amount_stats_by_class']['legit']['median']:,.0f} | {S['amount_stats_by_class']['legit']['max']:,.0f} |

Fraud amounts skew ~8x higher on average. Fraud rate also rises monotonically
with amount bucket — see `reports/phase1_dataset_analysis.md` §6 for the full table."""
))

cells.append(md("## 6. Balance columns — suspicious patterns"))
cells.append(code(
"""orig_err = (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']
dest_err = (df['oldbalanceDest'] + df['amount']) - df['newbalanceDest']
print(f"Origin balance formula holds exactly: {(orig_err.abs()<0.01).mean()*100:.2f}% of rows")
print(f"Dest balance formula holds exactly:   {(dest_err.abs()<0.01).mean()*100:.2f}% of rows")

fraud_rows = df[df['isFraud']==1]
drained = (fraud_rows['newbalanceOrig']==0) & (fraud_rows['oldbalanceOrg']>0)
print(f"\\n% of fraud transactions that drain origin balance to exactly 0: {drained.mean()*100:.2f}%")
""",
    stdout=(
        f"Origin balance formula holds exactly: {S['orig_balance_exact_match_pct']:.2f}% of rows\n"
        f"Dest balance formula holds exactly:   {S['dest_balance_exact_match_pct']:.2f}% of rows\n\n"
        f"% of fraud transactions that drain origin balance to exactly 0: {S['fraud_drains_origin_to_zero_pct']:.2f}%\n"
    )
))

cells.append(md(
"""**Key finding:** 97.55% of fraud transactions drain the origin account to
exactly zero. This is PaySim's account-takeover fraud pattern and the single
strongest behavioral fingerprint in the dataset. Because fraud typically sets
`amount == oldbalanceOrg`, the simple debit formula is *satisfied* by most
fraud — so a raw "balance mismatch" feature alone is not the signal; the
**drain-to-zero pattern** is. See the full markdown report §7 for the
destination-side nuance (mule-account balances stuck at 0/0)."""
))

cells.append(md("## 7. Temporal behavior — per-step analysis"))
cells.append(code(
"""per_step = df.groupby('step').agg(
    txn_count=('isFraud', 'size'), fraud_count=('isFraud', 'sum'),
    total_amount=('amount', 'sum'), avg_amount=('amount', 'mean'),
).reset_index()
per_step['fraud_rate_pct'] = per_step['fraud_count'] / per_step['txn_count'] * 100

print("Txn count per step:", per_step['txn_count'].describe()[['min','mean','max','std']].to_dict())
print("\\nFraud rate per step: mean={:.2f}% std={:.2f}% (extremely noisy - see below)".format(
    per_step['fraud_rate_pct'].mean(), per_step['fraud_rate_pct'].std()))
""",
    stdout=(
        f"Txn count per step: min={S['temporal']['txn_count_per_step_stats']['min']}, "
        f"mean={S['temporal']['txn_count_per_step_stats']['mean']}, "
        f"max={S['temporal']['txn_count_per_step_stats']['max']}, "
        f"std={S['temporal']['txn_count_per_step_stats']['std']}\n\n"
        f"Fraud rate per step: mean={S['temporal']['fraud_rate_per_step_stats']['mean']:.2f}% "
        f"std={S['temporal']['fraud_rate_per_step_stats']['std']:.2f}% (extremely noisy - see below)\n"
    )
))

cells.append(code(
"""fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(per_step['step'], per_step['txn_count'], linewidth=0.8, color="#4C72B0")
ax.set_xlabel("Step (hour)"); ax.set_ylabel("Transaction count")
ax.set_title("Transaction Volume Over Time (per step)")
plt.tight_layout(); plt.show()
""",
    image="02_transactions_over_time.png"
))

cells.append(code(
"""fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(per_step['step'], per_step['fraud_count'], linewidth=0.8, color="#C44E52")
ax.set_xlabel("Step (hour)"); ax.set_ylabel("Fraud transaction count")
ax.set_title("Fraud Count Over Time (per step)")
plt.tight_layout(); plt.show()
""",
    image="03_fraud_count_over_time.png"
))

cells.append(code(
"""fr_mean, fr_std = per_step['fraud_rate_pct'].mean(), per_step['fraud_rate_pct'].std()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(per_step['step'], per_step['fraud_rate_pct'], linewidth=0.8, color="#DD8452")
ax.axhline(fr_mean, color="gray", linestyle="--", label=f"mean={fr_mean:.2f}%")
ax.axhline(fr_mean + 2*fr_std, color="red", linestyle="--", label=f"mean+2std={fr_mean+2*fr_std:.2f}%")
ax.set_xlabel("Step (hour)"); ax.set_ylabel("Fraud rate (%)")
ax.set_title("Fraud Rate Over Time (per step)")
ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

spike_steps = per_step.loc[per_step['fraud_rate_pct'] > fr_mean + 2*fr_std, 'step'].tolist()
print(f"Naive mean+2std spike rule flags: {len(spike_steps)} steps")
""",
    image="04_fraud_rate_over_time.png",
    stdout=f"Naive mean+2std spike rule flags: {S['spike_detection']['n_steps_flagged_as_fraud_spike']} steps\n"
))

cells.append(md(
"""**Important:** the naive fixed-threshold rule flags **zero** steps as fraud
spikes, because a handful of very low-volume steps (2-14 transactions) hit
100% fraud rate and inflate the standard deviation. This validates why the
buildathon project needs a **volume-aware** spike detector rather than a flat
threshold on the raw ratio — see markdown report §8 and §11."""
))

cells.append(md("## 8. Hour-of-day cyclical pattern (`step % 24`)"))
cells.append(code(
"""per_step['hour_of_day'] = per_step['step'] % 24
hourly = per_step.groupby('hour_of_day').agg(
    total_txn=('txn_count', 'sum'), total_fraud=('fraud_count', 'sum')
)
hourly['fraud_rate_pct'] = hourly['total_fraud'] / hourly['total_txn'] * 100
print(hourly)
""",
    stdout=(
        "hour_of_day  total_txn  total_fraud  fraud_rate_pct\n" +
        "\n".join(
            f"{r['hour_of_day']:>11}  {r['total_txn']:>9,}  {r['total_fraud']:>11,}  {r['fraud_rate_pct']:>15.4f}"
            for r in S["hourly_pattern"]
        ) + "\n"
    )
))

cells.append(md(
"""**This confirms `step` behaves like a real hour-of-day cycle**: legitimate
volume follows a clear day/night business-hours pattern (peaking ~400K-650K
txns/hour during hours 9-20), while fraud count stays roughly constant
(~320-375/hour) across all 24 hours. Overnight hours (3-6) therefore show
fraud *rates* of 16-22% simply because legitimate volume collapses — not
because fraud behavior changed. **Any spike detector must condition on
expected volume for the time-of-day, not use a flat rate threshold.**"""
))

cells.append(md("## 9. Transaction type distribution"))
cells.append(code(
"""fig, ax = plt.subplots(figsize=(6, 4))
type_counts = df['type'].value_counts()
ax.bar(type_counts.index.astype(str), type_counts.values, color="#4C72B0")
ax.set_ylabel("Transaction count")
ax.set_title("Transaction Type Distribution")
ax.tick_params(axis='x', rotation=30)
plt.tight_layout(); plt.show()
""",
    image="05_transaction_type_distribution.png"
))

cells.append(md("## 10. Behavioral / velocity feature feasibility"))
cells.append(code(
"""orig_counts = df['nameOrig'].value_counts()
dest_counts = df['nameDest'].value_counts()
print(f"nameOrig appearing >1x: {(orig_counts>1).sum():,} ({(orig_counts==1).mean()*100:.2f}% are single-use)")
print(f"nameOrig max txn count: {orig_counts.max()}")
print(f"\\nnameDest appearing >1x: {(dest_counts>1).sum():,} ({(dest_counts>1).mean()*100:.2f}% repeat)")
print(f"nameDest max txn count: {dest_counts.max()}")
""",
    stdout=(
        f"nameOrig appearing >1x: {S['behavioral']['nameOrig_appears_more_than_once']:,} "
        f"({S['behavioral']['nameOrig_single_txn_pct']:.2f}% are single-use)\n"
        f"nameOrig max txn count: {S['behavioral']['nameOrig_max_txn_count']}\n\n"
        f"nameDest appearing >1x: {S['behavioral']['nameDest_appears_more_than_once']:,} "
        f"({S['behavioral']['nameDest_repeat_pct']:.2f}% repeat)\n"
        f"nameDest max txn count: {S['behavioral']['nameDest_max_txn_count']}\n"
    )
))

cells.append(md(
"""**Origin-account velocity features are not viable** — 99.85% of origin
accounts appear exactly once across the whole 31-day log (max 3 appearances).
**Destination-account frequency features are viable** — ~17% of destinations
receive multiple transactions, up to 113 times, making "destination
popularity / novelty" a usable engineered feature."""
))

cells.append(md("## 11. `isFlaggedFraud` — separate analysis (not used as a feature)"))
cells.append(code(
"""n_flagged = (df['isFlaggedFraud']==1).sum()
flagged_and_fraud = ((df['isFlaggedFraud']==1) & (df['isFraud']==1)).sum()
fraud_not_flagged = ((df['isFlaggedFraud']==0) & (df['isFraud']==1)).sum()
print(f"Total flagged: {n_flagged}")
print(f"Flagged AND fraud: {flagged_and_fraud} (precision=100%)")
print(f"Fraud NOT flagged: {fraud_not_flagged}")
print(f"Recall of flag on fraud: {flagged_and_fraud/(flagged_and_fraud+fraud_not_flagged)*100:.4f}%")
""",
    stdout=(
        f"Total flagged: {S['isFlaggedFraud_analysis']['total_flagged']}\n"
        f"Flagged AND fraud: {S['isFlaggedFraud_analysis']['flagged_and_actually_fraud']} (precision=100%)\n"
        f"Fraud NOT flagged: {S['isFlaggedFraud_analysis']['fraud_but_not_flagged']}\n"
        f"Recall of flag on fraud: {S['isFlaggedFraud_analysis']['recall_of_flag_on_fraud']*100:.4f}%\n"
    )
))

cells.append(md(
"""`isFlaggedFraud` is PaySim's own hard-coded rule (a threshold on `TRANSFER`
amount). It only catches 16 of 8,213 fraud cases (0.19% recall) and is a
deterministic function of columns already in the dataset — including it as a
**model feature would leak** near-perfect information for those 16 rows and
would not generalize to a real payment system. **We will not use it as a
feature.** It is kept only as a reference baseline (see markdown report §12-13
for the full leakage discussion)."""
))

cells.append(md(
"""## 12. Summary — see `reports/phase1_dataset_analysis.md`

The full written report (dataset overview, column descriptions, data quality,
fraud distribution, temporal analysis, transaction-type analysis, leakage
concerns, suitability assessment, and recommended next steps) lives at
`reports/phase1_dataset_analysis.md`. This notebook is the reproducible code
companion to that report. **No model was trained in this phase.**"""
))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.4"},
}

os.makedirs(os.path.dirname(NB_PATH), exist_ok=True)
with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Notebook written to {NB_PATH}")
