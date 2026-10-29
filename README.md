# BNPL Credit Screener

An academic prototype for **FINS5557 Applied AI in Finance** (Track A).
It scores the default risk of **thin-file** Buy-Now-Pay-Later applicants with a
gradient-boosting model and explains every decision, so a risk officer sees not
just a score but *why*.

> ⚠️ **Prototype only — not a real lending tool.** Do not enter real personal
> data. Decisions are illustrative and must not be used for actual credit calls.

## Features

- Default-risk score for a single applicant from an easy input form.
- Two models trained and compared: an interpretable **Logistic Regression**
  baseline and a stronger **HistGradientBoosting** main model.
- **Per-decision explanation** (SHAP when installed; a model-agnostic local
  fallback otherwise), tied to the applicant's own inputs.
- A **cost-sensitive decision threshold** (a missed defaulter costs more than a
  wrongly rejected good applicant), not a naive 0.5 cut-off.
- Input validation and graceful error handling.

## Project structure

```
bnpl-credit-screener/
├── app.py              # Streamlit app (UI, scoring, explanation)
├── requirements.txt
├── .gitignore          # excludes data + .env
├── .env.example
├── src/
│   ├── config.py       # paths, exposed input features, cost ratio
│   ├── train.py        # trains + evaluates + saves artifacts
│   └── predict.py      # load artifacts, score, explain
├── models/             # trained artifacts (committed so the app runs)
└── data/               # dataset goes here — NOT committed
```

## Dataset

Home Credit Default Risk (Kaggle).
<https://www.kaggle.com/competitions/home-credit-default-risk/data>

The dataset is **not** included in this repository — Kaggle's competition rules
do not allow redistribution, and the files are large. To reproduce:

1. Download from Kaggle (needs a Kaggle account; accept the competition rules).
2. Unzip `application_train.csv` into `data/`.

This prototype uses only the main `application_train.csv` table
(307,511 rows, 122 columns). Joining the auxiliary tables (bureau, previous
applications, etc.) is a planned improvement.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Train the models** (writes artifacts to `models/`):

```bash
python -m src.train --data data/application_train.csv
```

**2. Run the app:**

```bash
streamlit run app.py
```

Open the local URL Streamlit prints, fill in the applicant form, and click
**Score applicant**. You get a default probability, an approve / decline
decision against the cost-sensitive threshold, and a ranked explanation of the
features driving that applicant's score.

### Worked example

With the default form values (income 150,000; credit 500,000; age 40;
external score 2 = 0.5) the app returns a low-risk **APPROVE**. Lowering the
external credit scores toward 0 pushes the probability up and flips the
decision — a quick way to see the explanation respond.

*(Add a screenshot here before submission.)*

## API key setup (optional LLM explanation layer)

If you add the optional plain-language explanation layer, keys are read from a
local `.env` file (see `.env.example`) — **never hard-coded**:

```
OPENAI_API_KEY=sk-...
```

On Streamlit Cloud, set the same values under **App → Settings → Secrets**
instead of committing them.

## Deployment (public URL)

1. Push this repo to **public GitHub**.
2. On [Streamlit Community Cloud](https://streamlit.io/cloud), create an app
   pointing at `app.py`.
3. Add any secrets under App → Settings → Secrets.

## Known issues & planned improvements

- Uses only the main application table; AUC (~0.76) should rise after joining
  the bureau / previous-application tables.
- The cost ratio (`COST_FN:COST_FP`) in `src/config.py` is a first estimate and
  should be calibrated with realistic BNPL loss figures.
- SHAP explanations are computed per request and can be slow on the free tier.

## AI tool disclosure

Generative AI (Claude Code) assisted with scaffolding the repository structure,
the Streamlit UI, and drafting this README. All AI-assisted output was reviewed
and revised by the team. *(Update this with the exact tasks before submission.)*

## Attribution

- Data: Home Credit Default Risk, Kaggle.
- Libraries: scikit-learn, SHAP, Streamlit, pandas, NumPy.
