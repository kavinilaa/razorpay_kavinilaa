# AI Risk Manager

I built this as a defense-only fraud scoring tool for merchants. A merchant has two bad
options when fraud looks like an ordinary payment: miss it and lose money, or flag so much
legitimate activity that real customers get annoyed. I wanted the system to score an individual
transaction, show a human why it was scored that way, and add a second, clearly labeled view of
unusual activity over time.

## Why I chose this problem

I picked AI Risk Manager because the PaySim data made the merchant problem concrete instead of
abstract. Fraud appears only in `TRANSFER` and `CASH_OUT`; there are zero fraud rows in
`PAYMENT`, `CASH_IN`, or `DEBIT`. More strikingly, `97.55%` of fraudulent transactions drain
the origin account to exactly zero. That is a strong, checkable signal for a classifier, but it
is also a warning: a model that looks nearly perfect here may simply be learning PaySim's fraud
injection pattern.

The system is strictly detection and scoring only. It flags risk and explains it for a human
reviewer; it does not block an account, reverse a transaction, move money, or take any autonomous
action. There is no offensive capability hiding behind the interface.

## What I built and how to run it

I turned the trained models into a failure-aware Python risk engine, wrapped it in a FastAPI
service, and built a small React interface around the real API. The main workflow scores one
transaction and returns a risk band, recommended action, model explanation, and any available
spike context. The spike workflow looks up a time step or analyzes supplied aggregates, keeping
the statistical and Isolation Forest signals separate instead of pretending they are one reliable
score. Missing models, bad inputs, and unavailable spike artifacts return explicit states rather
than fabricated results.

There is no Docker setup in the repository, so I run it locally with two terminals. From the
repository root:

```powershell
python -m pip install -r requirements.txt
$env:RISK_API_API_KEY = "local-dev-key-CHANGE-ME"
uvicorn backend.main:app --reload --port 8000
```

In a second terminal:

```powershell
Set-Location frontend
npm install
$env:VITE_API_KEY = "local-dev-key-CHANGE-ME"
npm run dev
```

The frontend opens at `http://localhost:5173`. The backend is at `http://localhost:8000` and
the health endpoint is `http://localhost:8000/api/v1/health`. The committed key is for local
development only; I would replace it before sharing the service and set the real frontend origin
in `RISK_API_CORS_ALLOW_ORIGINS`.

The important pieces are small enough to find quickly:

- Transaction risk model: the XGBoost classifier in `models/xgboost.joblib`, with leakage-safe
  feature engineering and SHAP explanations in `src/`.
- Spike detector: a statistical z-score rule plus an Isolation Forest signal over hourly
  aggregates, exposed as a secondary triage aid.
- API: FastAPI routes in `backend/` for health, model info, transaction scoring, and spike
  analysis, protected by a shared local API key except for health checks.
- Frontend: the React/Vite app in `frontend/`, with separate loading, validation, network,
  degraded, and successful-score states.

The current test numbers are real, not copied from an earlier write-up: `80 passed` in the
Python backend suite and `30 passed` across `7` frontend Vitest files. Run them with:

```powershell
python -m pytest tests/ -q
Set-Location frontend
npm test
```

## Demo video: [Watch the AI Risk Manager demo](https://your-video-link)


The README is the main project guide; the application and model folders contain everything needed
to run the prototype.

## The model choices I made

For transaction scoring, I chose XGBoost because I have real labels in this dataset:
`isFraud`. Supervised learning is justified when I can train against a defined outcome, and
XGBoost won the comparison on validation PR-AUC (`0.9993`) while producing the strongest overall
test result in this dataset. I restricted that classifier to `TRANSFER` and `CASH_OUT` without
losing any fraud cases, because the other transaction types contain none here.

I deliberately did not use a fancier learned sequence or anomaly model as the only spike
detector. The z-score rule is simple, explainable, and needs no training, so it gives me a
transparent baseline for unusual hourly volume. I added Isolation Forest only after checking
whether it contributed a useful second signal. It does add a different view, but the honest
result is that neither method catches spikes reliably enough to stand alone. I expose both as
secondary evidence rather than turning them into a confident-looking combined score.

I also use SHAP for per-transaction explanations because a risk number without a reason is not
very useful to a reviewer. The explanation describes the actual feature value and direction
returned by the model; it isn't a fixed sentence selected by the frontend.

## What broke while I was building it

The spike detector was the uncomfortable result. I built it, evaluated it against the injected
ground truth, and found recall of only `6.63%` for the statistical method and `9.69%` for
Isolation Forest across all `743` steps. On the held-out test window, recall fell to `1.92%`
and `3.85%`. Instead of hiding that behind low false-positive counts or adding complexity until
the chart looked better, I traced the misses. The root-cause analysis showed that fraud often
appears in a handful of transactions during an otherwise quiet period, not as a large volume
spike. I kept it as a low-recall triage and root-cause signal and said plainly that the
transaction classifier is doing the heavy lifting.

I also caught a bug in the SHAP wording. An early explanation assumed that a large deviation from
the expected balance formula meant risk. That is backwards for this dataset: fraudulent
transactions usually drain the account exactly, so they often match the formula. I fixed the
template to describe the observed balance fact rather than attach a convenient but false risk
story to it. That was a useful check that I was reading the model and the data together, not just
shipping the first explanation that sounded plausible.

The biggest limitation is still the dataset. This was trained and tested on synthetic PaySim
transactions, not real merchant traffic. The near-perfect classifier scores are a property of
PaySim's fraud-injection pattern, especially the account-drain signal; they are not a real-world
accuracy claim. A real deployment would need representative labeled data, monitoring for drift,
and threshold decisions based on the merchant's actual review capacity and customer cost.