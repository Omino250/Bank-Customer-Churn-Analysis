# 🏦 Bank Customer Churn Analysis & Machine Learning Prediction

> Turning 10,000 rows of transaction history into a $17.33M retention strategy.

---

## Project Overview

A retail bank was quietly bleeding customers — 1 in 5, to be exact. I dug into 10,000 customer records to find out who was leaving, why, and whether it could be caught before the account closed. It could. I built a Random Forest classifier that flags at-risk customers with 84% discriminative power, tuned it to catch 59% of churners in advance, and packaged the whole thing into a 3-page Power BI dashboard that hands retention teams a live, prioritized rescue list.

The headline number: **191 active accounts, worth $17.33M in deposits, are sitting in the risk pipeline right now** — and 53 of them are a very specific, very findable type of customer.

---

## Table of Contents
- [The Business Problem](#the-business-problem)
- [Tech Stack](#tech-stack--data-architecture)
- [The Workflow (Including the Parts That Broke)](#the-workflow-including-the-parts-that-broke)
- [The 10 Questions the Business Actually Asked](#the-10-questions-the-business-actually-asked)
- [Machine Learning: From Baseline to Deployable Model](#machine-learning-from-baseline-to-deployable-model)
- [Power BI: Turning a Model Into a Daily Habit](#power-bi-turning-a-model-into-a-daily-habit)
- [The Recommendations, All in One Place](#strategic-recommendations)
- [Repository Structure](#repository-structure)
- [How to Run This Project](#how-to-run-this-project)

---

## The Business Problem

Retail banking leadership had noticed something uncomfortable: more customers were closing accounts than the acquisition pipeline could replace. Nobody had a clear answer for *why*. Was it price? Service? A specific market? A specific age group? The account team had theories, but nothing backed by numbers.

So the mandate was simple to state and hard to execute:

1. **Find out why customers are leaving** — across demographics, financial behavior, product usage, and service history.
2. **Build an early-warning system** that flags at-risk accounts *before* they close, not after.
3. **Put it in the hands of the people who can act on it** — not buried in a notebook, but live in a dashboard retention teams open every morning.

Once the data was in, the scale of the problem became concrete fast: a **20.38% churn rate**, translating into **$185.68M in lost capital**, and — this is the part that should worry any CFO — the customers walking out the door carried an **average balance of $91.11K**, noticeably *higher* than the $76.5K portfolio average. This wasn't low-value account attrition. The bank was losing its best clients.

---

## Tech Stack & Data Architecture

| Layer | Tools |
|---|---|
| Data Processing & Analytics | Python (pandas, NumPy) |
| Machine Learning | Scikit-Learn (Random Forest Classifier, Feature Importance, ROC-AUC) |
| Visualization & BI | Power BI, DAX, Power Query |
| Storage | CSV, MySQL |
| Environment | Jupyter Notebook, VS Code, Python virtual environment (.venv) |

---

## The Workflow

Most project write-ups skip the messy middle. I'm keeping it in, because the setbacks shaped the final model as much as the clean wins did.

### 1. Data Cleaning & Feature Engineering
The raw dataset came in clean by banking standards — 10,000 rows, 18 columns, zero missing values. But "clean" doesn't mean "model-ready." I dropped identifier columns (`RowNumber`, `CustomerId`, `Surname`) that carry no predictive signal, standardized column naming (`HasCrCard` → `HasCreditCard`), and binned continuous variables — age, tenure, credit score, balance, salary, reward points — into operational brackets that map to how a retention team actually thinks about customers (e.g., "41–60, inactive, good credit" is a segment you can build a campaign around; "customer #4,812" is not).

One early gotcha worth flagging for anyone reusing this pipeline: binning `EstimatedSalary` and `Point Earned` with a fixed upper bound silently produces `NaN`s for anyone above that cap. The fix is trivial — cap the top bin at `np.inf` — but it's the kind of thing that will quietly corrupt a segment analysis if you don't check `isnull().sum()` after every binning step.

### 2. The Environment Rabbit Hole
Before any modeling happened, I lost time to something almost every data scientist has hit: a terminal that opened in the wrong directory, silently failed to activate the virtual environment, and installed `seaborn` into the global Python install instead of `.venv`. The notebook kept running against a kernel that couldn't see the packages I'd just installed. The fix was mechanical — `cd` into the project root, activate `.venv`, reinstall inside it, and explicitly select the `.venv` interpreter in the notebook's kernel picker — but it's a reminder that environment hygiene is part of the analysis, not separate from it.

### 3. The Target Leakage Trap
This was the most important mistake in the whole project, and I'm glad I caught it before it reached production.

The dataset included a `Complain` column. On the surface, it looked like a perfectly reasonable predictive feature — of course complaints matter to churn. But when I cross-tabulated it against the target:

- Customers **without** a complaint churned **0.05%** of the time.
- Customers **with** a complaint churned **99.51%** of the time.

That's not a strong predictor. That's a proxy for the target itself. A complaint, in this dataset, is filed as the *last step before leaving*, not an early warning sign the bank ever got a chance to act on. Feeding that into a model produces a classifier that looks brilliant on paper (99%+ accuracy) and is completely useless in production, because it has learned the rule "if `Complain == 1`, predict churn" and ignored every genuinely actionable signal — age, balance, product count, activity status.

**I dropped `Complain` from the feature set entirely.** The honest baseline accuracy dropped from a fake 99% down to a real **85.30%** — and that drop is exactly what tells you the leak was real. A model that gets *worse* when you remove a leaking feature is a model that was lying to you before.

### 4. Model Selection: Why Random Forest
Linear models assume relationships move in straight lines. This data doesn't cooperate:

- **Credit score** shows almost no difference in raw averages between churned (645) and retained (652) customers — but the distribution tells a different story, with churn risk concentrated almost entirely in a left-tail cluster below ~400.
- **Product holdings** follow a sharp J-curve: churn *drops* to 7.60% at 2 products, then *explodes* to 82.71% at 3 products and 100% at 4.

Those are threshold effects and interactions, not linear trends. A Random Forest classifier handles both natively — it splits on thresholds, captures multi-feature interactions without manual engineering, is robust to the outliers sitting in the credit score tail, and hands back feature importances that let me sanity-check the model against the EDA rather than treating it as a black box.

### 5. Tuning for the Metric That Actually Matters
The clean, post-leakage model hit 85.30% accuracy but only caught **46% of actual churners** (recall on the churned class) at the default 0.50 decision threshold. In banking terms: missing more than half your departing customers is a very expensive kind of "accurate."

I first tried `class_weight='balanced'`, expecting it to rebalance the model's attention toward the minority class. It barely moved the needle (recall went from 46% to 45%) — because Random Forest by default grows trees to near-pure leaves, so reweighting the loss during splitting has little effect once the leaves are already homogeneous.

The fix that actually worked was **threshold tuning**, not reweighting. Lowering the decision cutoff from 0.50 to **0.35** shifted the numbers to:

| Metric | @0.50 threshold | @0.35 threshold |
|---|---|---|
| Accuracy | 85.30% | 82.80% |
| Churn Recall | 46% | **59%** |
| Churn Precision | 72% | 58% |
| ROC-AUC | 0.84 | 0.84 (unchanged — threshold doesn't affect this) |

Yes, precision dropped — the model now flags some customers who were never going to leave. But the economics of churn are asymmetric: a false positive costs a low-value retention email or a small loyalty gesture. A false negative costs a $91K account. Trading 14 points of precision for 13 points of recall is a straightforward win once you price it that way.

---

## The 10 Questions the Business Actually Asked

Each question below follows the same diagnostic structure: **what happened, why it happened, and what to do next.**

### 1. Who are our churned customers? (Demographic Profile)
**What happened:** Churn is not evenly spread across demographics. The 41–60 age bracket drives **60.65% of all exits** (1,236 of 2,038 churned accounts), despite representing a smaller slice of the overall customer base than younger cohorts. Female customers churn at 25.07% versus 16.47% for male customers — a 52% relative gap.

**Why it happened:** Middle-aged customers typically carry the highest balances and the most complex product needs (mortgages, wealth products, multi-account relationships), which means they have the most to lose from poor service *and* the most attractive profile for competitor banks to court. The gender gap likely reflects differences in product fit or engagement with specific offerings, though the dataset alone can't isolate the exact mechanism.

**What to do next:** Build age-segmented and gender-aware retention playbooks rather than a single generic campaign. The 41–60 segment, specifically, warrants a dedicated relationship-management track — this is not a segment to treat with a mass-market email.

### 2. Is churn linked to satisfaction scores or complaints?
**What happened:** Self-reported satisfaction score has **zero predictive value** — churn sits flat between 19.64% and 21.80% regardless of whether a customer rated their experience a 1 or a 5. Complaints, on the other hand, are a near-perfect signal: 99.80% of churned customers had filed one.

**Why it happened:** Satisfaction surveys capture a moment-in-time sentiment that doesn't translate into behavior. Complaints, by contrast, are typically filed at the *end* of a dispute cycle — by the time one is logged, the decision to leave is usually already made.

**What to do next:** Stop relying on satisfaction surveys as an early-warning tool — they aren't one. Treat every logged complaint as a red-alert, same-day intervention trigger, not a ticket in a standard queue, because the data shows there's almost no runway left once a complaint is filed.

### 3. Do low engagement, inactivity, or lack of a credit card predict churn?
**What happened:** Inactive members churn at **26.87%** versus **14.27%** for active members — activity nearly halves churn risk. Credit card ownership, by contrast, makes almost no difference (20.81% vs. 20.20%).

**Why it happened:** Engagement is a behavioral signal that precedes a decision; simply holding a card is a static attribute that doesn't reflect usage. A customer can own a credit card and still be mentally checked out of the relationship.

**What to do next:** Build automated re-engagement triggers keyed to activity drop-off, not card ownership. Retire "issue more cards" as a retention lever — the data doesn't support it.

### 4. How do balance, salary, tenure, and credit score affect loyalty?
**What happened:** Balance matters a lot — Medium and High balance tiers churn at ~24%, versus 14.25% for Low-balance accounts. Salary shows almost no variation (19.87%–20.23% across all tiers). Tenure shows a shallow, weak trend (21.15% at 0–2 years down to 19.69% at 6–10 years). Credit score shows a threshold effect: flat until the "Poor" tier, which spikes to 32.28%.

**Why it happened:** Funded accounts (balance) represent customers with real switching options — they have the capital to move, and competitors have the incentive to court them. Salary is a proxy for income, not banking behavior, which is why it shows no signal. Tenure alone doesn't build enough switching friction to meaningfully protect a relationship. Credit score risk is concentrated at the extreme low end, likely tied to financial distress rather than general creditworthiness.

**What to do next:** Prioritize retention spend by balance tier, not tenure or income. Long-standing customers are not automatically "safe" and shouldn't be deprioritized just because they're loyal on paper.

### 5. Which card types drive churn, and does point accumulation help?
**What happened:** Card type churn is nearly uniform (19.26%–21.78% across Gold, Silver, Platinum, Diamond). Reward points show the same flat pattern (19.86%–21.04% across Low/Medium/High point tiers) — and points don't even scale with tenure; long-standing customers earn points at the same rate as new sign-ups.

**Why it happened:** Premium card tiers aren't currently differentiated enough in perceived value to change customer behavior, and the loyalty program isn't structured to reward — or retain — long-term relationships.

**What to do next:** Audit the rewards program. It's not doing its job as a retention lever. Consider restructuring point accrual to scale with tenure and redemption value to scale with relationship depth, rather than flat per-transaction accumulation.

### 6. Can we build an early-detection risk model?
**What happened:** Yes. A Random Forest classifier trained on leakage-free features (Age, Credit Score, Balance, Number of Products, Active Status) achieves an **84% ROC-AUC** and, at a tuned 0.35 threshold, catches **59% of churners** in advance with 58% precision.

**Why it happened:** Once the fake signal (`Complain`) was removed, the model was forced to learn from genuine pre-churn behavioral and financial patterns — the same ones surfaced in the EDA above.

**What to do next:** Deploy the model's probability scores into a live operational pipeline (see the Power BI section below) so retention teams work from a ranked list instead of a hunch.

---

## Machine Learning: From Baseline to Deployable Model

**Feature Importance** (what the model actually relies on):

| Feature | Importance |
|---|---|
| Age | 28.6% |
| Credit Score | 25.3% |
| Balance | 21.1% |
| Number of Products | 13.2% |
| Satisfaction Score | 6.0% |
| Active Member | 3.9% |
| Has Credit Card | 1.6% |

The top four features account for roughly **88%** of the model's decision-making — and every one of them lines up with a pattern already surfaced independently in the EDA. That agreement between "what the statistics show" and "what the model learned" is the strongest evidence the pipeline is measuring something real, not overfitting to noise.

**Final model performance at the 0.35 operating threshold:**
- ROC-AUC: **0.84**
- Recall (Churned): **59%**
- Precision (Churned): **58%**
- False Positive Rate: **11%**

That combination — catching 6 in 10 departing customers while only flagging 11% of stable customers unnecessarily — is the sweet spot the ROC curve's "knee" pointed to, and it's the threshold shipped into the dashboard below.

---

## Power BI: Turning a Model Into a Daily Habit

A model sitting in a notebook doesn't save anyone's account. The three dashboard pages below are what actually gets opened every morning.

### Page 1 — Customer Overview
<img width="1059" height="598" alt="image" src="https://github.com/user-attachments/assets/ab46700c-11df-4560-be3c-3f331d641b1a" />


**What happened:** The portfolio holds 10,000 accounts, $764.86M in total deposits, and a 51.51% active engagement rate. France anchors over half of total volume. Nearly 70% of customers carry a "Good" credit score. Reward points are flat across every tenure bracket.

**Why it happened:** This page is the health check — it's not diagnosing churn yet, it's establishing that the underlying book of business is fundamentally sound (good credit quality, strong deposits), which makes the churn losses on the next page even more costly, because they're not "bad" customers leaving.

**What to do next:** Use this page as the baseline reference point in every retention review — it's the "before" picture the churn numbers should be measured against.

### Page 2 — Historical Churn Analysis
<img width="1064" height="598" alt="image" src="https://github.com/user-attachments/assets/6d4a6cc2-4a41-4668-9511-6f0354b01fb3" />


**What happened:** 2,038 accounts churned — a 20.38% rate — for a $185.68M capital loss, at an average lost balance of $91.11K per account (above the portfolio average). Germany (814 accounts, 39.94%) and France (811 accounts, 39.79%) together account for roughly 80% of total churn volume. The 41–60 age group drives 60.65% of exits. 99.80% of churned customers had a logged complaint.

**Why it happened:** Germany losing an equivalent volume to France despite having roughly half the customer base confirms Germany isn't just "unlucky" — it has a structurally higher churn *rate*, pointing to market-specific friction (competition, fees, service delivery) rather than a global product failure. If this were a universal issue, churn would be flat across geographies; it isn't.

**What to do next:** Treat Germany as its own retention project, not a line item in a global campaign. Route dedicated budget and a root-cause investigation (pricing, local competition, service quality) specifically at the German book. Simultaneously, since France carries the largest absolute volume, even a modest improvement in French retention outperforms an equivalent effort anywhere else in the portfolio.

### Page 3 — Churn Prediction & Risk Pipeline
<img width="1057" height="593" alt="image" src="https://github.com/user-attachments/assets/427a341e-2140-408d-b3ea-2251449740fe" />


**What happened:** The model currently flags **191 active accounts** — $17.33M in deposits — as at risk, split into a Medium Risk cohort (149 accounts) and a High Risk cohort (42 accounts, the top 28.2% requiring immediate action). Drilling into the decomposition tree: 136 of the 191 (71.2%) are aged 41–60. Of those, 77 are inactive. Of those 77, **53 hold a "Good" credit score.**

**Why it happened:** This is the same story as Page 2, but forward-looking instead of historical — the model has essentially found the *next* wave of the exact pattern that already cost the bank $185.68M. The inactivity-before-exit pattern from the EDA (engaged members churn at half the rate of inactive ones) shows up again here as the dominant path through the tree.

**What to do next:** This is the actionable core of the whole project — the **53 middle-aged, high-credit, inactive customers** are the highest-leverage retention target the bank has. They're not credit-risk cases the bank should be relieved to lose; they're prime clients disengaging quietly before they walk. Front-line teams should work this exact segment first, using the ranked Operational Rescue List (sorted by individual risk score, some cases as high as 86–91% probability) as a daily call sheet, filtered above the 0.35 threshold.

---

## Strategic Recommendations

Pulling every "what to do next" above into one prioritized list:

1. **Stand up the 53-customer rescue list as a live workflow, this week.** Middle-aged (41–60), inactive, good-credit customers are the highest-value, most-findable at-risk segment in the portfolio. This is the single highest-ROI action available from this project.
2. **Make Germany a standalone retention initiative**, not a footnote in a global plan — its churn rate is roughly double France's and Spain's, and that gap won't close with generic tactics.
3. **Redesign the cross-sell playbook around the product J-curve.** Push single-product customers toward a second product (churn drops from 27.71% to 7.60%), but build a hard governance gate that stops sales teams from pushing customers to 3+ products, where churn spikes to 82.71%–100%.
4. **Escalate every logged complaint as same-day, high-severity.** With a 99.80% correlation to churn, there's essentially no time left to react once one is filed — the intervention has to happen earlier, at the activity-drop-off stage, not the complaint stage.
5. **Retire satisfaction-survey monitoring and reward-point volume as churn signals.** Both are statistically flat against the outcome. Redirect that monitoring effort toward activity status and balance-tier movement, which actually separate the classes.
6. **Operationalize the 0.35-threshold model output directly into CRM task queues**, ranked by probability score, so retention capacity is spent on the accounts most likely to actually leave — not spread evenly across the base.

---

## Repository Structure

```
├── data/
│   ├── raw_bank_data.csv
│   └── bank_churn_scored_secure.csv
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   └── 02_model_training_evaluation.ipynb
├── reports/
│   └── bank_churn_dashboard.pbix
├── README.md
└── requirements.txt
```

## How to Run This Project

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/bank-churn-prediction.git
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the model pipeline**
Execute the notebooks in `notebooks/` in order — `01_data_cleaning_eda.ipynb` first (cleaning, feature engineering, EDA), then `02_model_training_evaluation.ipynb` (leakage removal, model training, threshold tuning, feature importance).

**4. Open the dashboard**
Open `reports/bank_churn_dashboard.pbix` in Power BI Desktop to explore the interactive visuals, the decomposition tree, and the Operational Rescue List.

---

