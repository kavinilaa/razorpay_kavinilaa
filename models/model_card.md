# Model Card — Transaction-Level Fraud Risk Model

**Project:** AI Fraud-Spike & Risk Detection System (Razorpay Buildathon, Track 02)
**Model file:** `models/xgboost.joblib` · **Metadata:** `models/model_metadata.json`
**Version:** trained with `random_seed=42`, selected by validation PR-AUC (see
`reports/phase3_model_comparison.md`)

> **This is a prototype built for a hackathon/buildathon submission, demonstrated entirely on a
> synthetic dataset (PaySim). It must NOT be described, marketed, or deployed as a
> production-ready fraud detector trained on real Razorpay transaction data. It has never seen
> real payment data of any kind.**

---

## 1. Purpose

Score a single TRANSFER or CASH_OUT transaction with a probability that it is fraudulent, to
support a human review workflow (risk band + recommended action + explanation), as part of a
larger fraud-spike & risk detection system that also includes an aggregate time-series
spike-detection layer (`reports/phase3_spike_detection.md`).

## 2. Dataset

- **Source:** PaySim mobile-money simulation log (`data/raw/PS_20174392719_1491204439457_log.csv`),
  a synthetic dataset — see §11.
- **Size:** 6,362,620 transactions, 11 raw columns (`reports/phase1_dataset_analysis.md`).
- **Modeling universe:** restricted to `TRANSFER` + `CASH_OUT` transactions
  (2,770,409 rows, 0.2965% fraud rate) — this retains **100% of fraud cases** while excluding
  `PAYMENT`/`CASH_IN`/`DEBIT`, which are structurally fraud-free in this dataset
  (`reports/phase3_model_comparison.md` §1).
- **Class imbalance:** ~337:1 within the restricted universe (774:1 on the full dataset).

## 3. Training / Validation / Test periods

A strict chronological split on `step` (PaySim's simulated hour, 1–743 ≈ 31 simulated days) —
**never a random split** — so validation/test never see transactions from before they occurred
relative to training:

| Split | Steps | Rows | Fraud rows | Fraud rate | Imbalance |
|---|---|---|---|---|---|
| **Train** | 1–520 | 2,653,729 | 5,781 | 0.2178% | 458.0 : 1 |
| **Validation** | 521–631 | 78,701 | 1,180 | 1.4993% | 65.7 : 1 |
| **Test** | 632–743 | 37,979 | 1,252 | 3.2966% | 29.3 : 1 |

Fraud rate rises across splits because fraud is not stationary over the timeline — this is why
the split is chronological, to honestly measure generalization forward in time rather than
interpolation.

## 4. Features (23)

`amount`, `log_amount`, `balance_change_orig`, `balance_error_orig`, `drained_to_zero`,
`origin_balance_zero_before`, `amount_to_origin_balance_ratio`, `balance_change_dest`,
`balance_error_dest`, `dest_balance_artifact_flag`, `hour_of_day`, `hour_sin`, `hour_cos`,
`is_transfer`, `hist_dest_txn_count`, `hist_dest_unique_origin_count`, `hist_dest_total_amount`,
`hist_dest_avg_amount`, `hist_dest_max_amount`, `destination_is_new`, `amount_to_dest_avg_ratio`,
`amount_to_type_avg_ratio`, `high_amount_indicator`.

Full definitions: `reports/phase2_feature_dictionary.md`. All are leakage-safe by construction —
destination-history features use a strict "steps < T" expanding window
(`src/features/feature_engineering.py::compute_destination_history`), never full-dataset stats
computed retroactively.

## 5. Excluded features (and why)

| Feature | Reason excluded |
|---|---|
| `isFlaggedFraud` | PaySim's own naive rule; circular, near-zero standalone signal (AUC 0.50), 0.19% recall on its own |
| `nameOrig`, `nameDest` (raw strings) | High-cardinality IDs — 99.85% of `nameOrig` values are single-use; pure memorization risk |
| Raw `oldbalanceOrg`/`newbalanceOrig`/`oldbalanceDest`/`newbalanceDest` | Only 20–34% formula-consistent due to PaySim simulation quirks — engineered discrepancy features used instead |
| Raw `step` / `day_index` | Ordinal time position, excluded from the classifier to avoid keying on "which chunk of the 743-step timeline" rather than a generalizable pattern (`hour_of_day` + cyclic `hour_sin`/`hour_cos` retain the legitimate time-of-day signal) |
| Raw `type` string | Redundant with `is_transfer` once the universe is restricted to 2 types |
| `eligible_for_fraud_model` | Constant 1 after filtering — no information |

## 6. Target

`isFraud` (binary, ground truth from the PaySim simulator).

## 7. Metrics

Three models were trained and compared (`reports/phase3_model_comparison.md`); **XGBoost was
selected** by highest validation PR-AUC.

| Model | Val PR-AUC | Val ROC-AUC | Test PR-AUC | Test ROC-AUC | Test Precision | Test Recall | Test FP | Test FN |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9342 | 0.9980 | 0.9542 | 0.9978 | 0.5156 | 0.9880 | 1,162 | 15 |
| LightGBM | 0.9992 | 0.9992 | 0.9992 | 0.9993 | 0.9929 | 0.9992 | 9 | 1 |
| **XGBoost (selected)** | **0.9993** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.9992** | **0** | **1** |

At the default 0.5 threshold, test-set XGBoost: 1,251/1,252 fraud caught, 0 false positives out
of 36,727 legitimate transactions.

## 8. Threshold strategy

No fixed monetary cost ratio is available for this project (a documented open item —
`reports/phase2_modeling_recommendation.md` §8), so three **conceptual operating modes** are
defined from a full threshold sweep on validation (`reports/phase3_threshold_analysis.md`), then
confirmed once on test:

| Mode | Threshold | Validation P / R | Test P / R | Alert rate |
|---|---|---|---|---|
| HIGH_RECALL | 0.43 | 1.000 / 0.999 | 1.000 / 0.999 | ~1.50% |
| BALANCED | 0.43 | 1.000 / 0.999 | 1.000 / 0.999 | ~1.50% |
| HIGH_PRECISION | 0.17 | 0.901 / 0.999 | 0.965 / 1.000 | ~1.66% |

HIGH_RECALL and BALANCED land on the same threshold because validation scores are extremely
well-separated for this dataset (see the threshold analysis report for why) — this is a property
of this dataset, not a guarantee that will hold for any other. Risk bands (`LOW`/`MEDIUM`/`HIGH`)
used by `src/risk_engine/thresholds.py` are built directly from these two measured thresholds
(0.17, 0.43), not an arbitrary split.

## 9. Known limitations

- **Near-perfect test metrics are a dataset artifact, not a generalization claim** (see §11).
- Model behavior on transaction types outside TRANSFER/CASH_OUT is untested and unsupported by
  design (this model should never be asked to score a PAYMENT/CASH_IN/DEBIT transaction).
- One fraud case in the validation set was missed at every threshold from 0.13–0.89 — its
  features apparently resemble legitimate activity to the model regardless of cutoff.
- Destination-history features (`hist_dest_*`) require a live history lookup in production; the
  standalone `predict_transaction_risk()` function defaults an unknown destination to "brand
  new" when the caller doesn't supply real history — this is a conservative default, not a real
  lookup (see `reports/provenance.md` and `src/risk_engine/risk_engine.py`).
- Threshold/band boundaries are validated on this dataset's fraud pattern only, not against any
  real-world cost ratio.

## 10. Synthetic-data limitation (read this before quoting any metric above)

PaySim's fraud-injection design makes **97.55% of fraud drain the origin account's balance to
exactly zero**, and this transaction almost always satisfies the basic debit-balance formula
exactly in the process (`reports/phase1_dataset_analysis.md` §7). This single, easily-engineered
behavioral fingerprint (`drained_to_zero` + `balance_error_orig` ≈ 0) is enough to separate
nearly all fraud from non-fraud in this dataset — confirmed as the top feature-importance signal
(`reports/phase3_model_comparison.md` §8). **Real-world fraudsters do not reliably reproduce this
exact signature, and legitimate high-value transfers essentially never produce it in this
dataset — but that is a property of the simulator, not of real payment fraud.** The near-perfect
PR-AUC/precision/recall figures in §7 demonstrate that the modeling pipeline is leakage-free and
correctly calibrated to *this* dataset's mechanic; they are not evidence this exact model would
achieve anything close to this performance against real, adversarial fraud patterns.

## 11. Leakage controls

1. Chronological (not random) train/val/test split (§3).
2. Destination-history features computed with a strict "steps < T" expanding window — never
   full-dataset stats attached retroactively (`reports/phase2_feature_leakage_audit.md`).
3. `scale_pos_weight` for LightGBM/XGBoost computed strictly from the TRAIN partition only.
4. `high_amount_indicator`'s per-type threshold frozen from a training-period slice (steps ≤ 594,
   80% of the range), applied unchanged to all later rows — mirroring how a deployed threshold
   would be fixed at training time.
5. The test set was scored exactly once, at the end, for final reporting
   (`src/models/train_transaction_models.py` module docstring).
6. Single-feature AUC audit performed on all engineered features to check for accidental
   near-target-equivalent features — none found (`reports/phase2_feature_leakage_audit.md` §6).

## 12. Intended use

- A prototype/demo component of a buildathon submission, showing a leakage-safe, explainable,
  time-split-validated approach to transaction fraud scoring on a public synthetic benchmark.
- A reference implementation for how such a system's failure handling, explainability, and
  threshold strategy could be structured (`src/risk_engine/`), pending real-data validation.

## 13. Prohibited use

- **Must not be presented as, or deployed as, a production-ready fraud detector trained on real
  Razorpay (or any real merchant) transaction data.** It has never been trained or validated on
  real payment data.
- Must not be used to make final automated block/allow decisions on real transactions without a
  human review step and real-data revalidation.
- Must not be applied to transaction types outside TRANSFER/CASH_OUT, or presented as detecting
  fraud patterns other than the account-takeover/full-drain pattern this specific dataset
  encodes.
- Must not have its test-set metrics (§7) quoted without the synthetic-data caveat (§10)
  alongside them.
