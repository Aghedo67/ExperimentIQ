# ExperimentIQ — A/B Test Design & Causal Analysis Platform

## Business Problem

The Product team at MobilePlay UK wants to test whether moving a friction gate in their mobile game from level 30 to level 40 improves 7-day player retention. Before looking at any outcome data, the analytics team must design the experiment correctly — calculate the required sample size, define the minimum detectable effect, and pre-register the analysis plan. Only then can results be trusted.

---

## Project Structure

```
ExperimentIQ/
├── data/
│   ├── raw/                          # Downloaded dataset (not in Git)
│   └── processed/                    # Cleaned and annotated data
├── notebooks/
│   ├── 01_experiment_design.ipynb    # Power analysis, MDE, sample size, randomisation check
│   ├── 02_core_analysis.ipynb        # Hypothesis testing, CIs, segmentation, Simpson's paradox
│   ├── 03_advanced_methods.ipynb     # Bootstrap, CUPED variance reduction, sequential testing
│   └── 04_recommendation.ipynb       # Decision framework, expected value, rollout recommendation
├── app.py                            # Interactive Streamlit dashboard
├── outputs/
│   └── figures/                      # Saved charts
├── requirements.txt
└── README.md
```

---

## Key Findings

*(Populated after notebooks are run)*

| Metric | Finding |
|--------|---------|
| 7-day retention — gate_30 (control) | TBD |
| 7-day retention — gate_40 (treatment) | TBD |
| Observed lift | TBD |
| Statistical significance | TBD |
| CUPED variance reduction | TBD |
| Business recommendation | TBD |

---

## Live Dashboard

*(Link added after Streamlit deployment)*

---

## Technical Stack

| Layer | Tools |
|-------|-------|
| Data processing | Python, pandas, numpy |
| Statistical inference | scipy (z-test, bootstrap, power analysis) |
| Variance reduction | CUPED (manual implementation) |
| Sequential testing | Alpha-spending / O'Brien-Fleming bounds |
| Visualisation | matplotlib, seaborn, plotly |
| Dashboard | Streamlit |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (see data/raw/README.md)

# 3. Run notebooks in order
jupyter notebook notebooks/01_experiment_design.ipynb

# 4. Launch dashboard
streamlit run app.py
```

---

## Data Source

[Cookie Cats A/B Test](https://www.kaggle.com/datasets/yufengsui/mobile-games-ab-testing) — 90,189 players from the mobile game Cookie Cats (Tactile Entertainment), randomised between gate placement at level 30 (control) and level 40 (treatment).
