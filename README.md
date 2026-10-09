# 🏦 Bank Customer Churn Analysis & Machine Learning Prediction

> Turning 10,000 rows of transaction history into a $73.75M retention strategy.

---

## Introduction

A retail bank was quietly bleeding customers, specifically 1 in 5. I analyzed 10,000 customer records to determine who was leaving, why, and whether we could catch them before their accounts closed. 

I built an XGBoost classifier with an 87.5% ROC-AUC score, engineered features like balance-to-salary ratios, and tuned hyperparameters using GPU-accelerated RandomizedSearchCV. By shifting the decision threshold from 0.50 down to 0.2909 based on Precision-Recall curve analysis, the model catches 69% of churners in advance with 59% precision.

The entire system feeds into a 3-page Power BI dashboard that delivers a live, prioritized operational rescue list. The headline number: **798 active accounts carrying $73.75M in deposits sit in the risk pipeline right now**, with 581 of them concentrated in a high-value middle-aged demographic.

---

## Table of Contents
- [The Business Problem](#the-business-problem)
- [Tech Stack & Data Architecture](#tech-stack--data-architecture)
- [The Workflow](#the-workflow)
- [The 10 Questions the Business Actually Asked](#the-10-questions-the-business-actually-asked)
- [Machine Learning: From Baseline to Deployable Model](#machine-learning-from-baseline-to-deployable-model)
- [Power BI: Turning a Model Into a Daily Habit](#power-bi-turning-a-model-into-a-daily-habit)
- [Strategic Recommendations](#strategic-recommendations)
- [Repository Structure](#repository-structure)
- [How to Run This Project](#how-to-run-this-project)

---

## The Business Problem

Retail banking leadership noticed customer account closures were outpacing new account acquisition. Leadership had theories about pricing, service quality, and regional markets, but lacked numerical validation.

The objective was straightforward:

1. **Identify churn drivers** across demographics, financial behavior, product usage, and service history.
2. **Build an early warning system** that flags at-risk accounts before they close.
3. **Operationalize insights** into an interactive dashboard for daily retention workflows.

The initial diagnostic revealed a **20.38% churn rate**, representing **$185.68M in lost capital**. Departing customers held an **average balance of $91.11K**, significantly higher than the $76.5K portfolio average. The bank was losing its highest-value clients.

---

## Tech Stack & Data Architecture

| Layer | Tools |
|---|---|
| Data Processing & Analytics | Python (pandas, NumPy) |
| Machine Learning | XGBoost, Scikit-Learn (RandomizedSearchCV, ROC-AUC, PR Curves) |
| Visualization & BI | Power BI, DAX, Power Query |
| Storage | CSV, MySQL |
| Environment | Jupyter Notebook, VS Code, Python virtual environment (.venv) |

---

## The Workflow

Here's the thing about real data science projects: the technical hurdles along the way shape the final solution as much as the clean wins do.

### 1. Data Cleaning & Feature Engineering
The raw dataset contained 10,000 rows and 18 columns with zero missing values. To improve model performance, I removed non-predictive identifiers (RowNumber, CustomerId, Surname) and standardized column names. 

Raw metrics like balance and age only tell half the story. I engineered domain-specific interaction features, including **balance-to-salary ratio** and **credit per age**. A high balance means something completely different for a low-income earner compared to a high-income earner. Continuous variables were also binned into operational brackets that align with how retention teams build targeted campaigns.

### 2. The Data Leakage Trap
During initial model evaluation, the baseline returned a perfect 1.0 ROC-AUC score. A perfect score is a clear sign that a model is cheating. 

Inspecting feature importances revealed that target proxy columns (including Actual_Churn, Predicted_Churn, and Complain) had been left in the training dataset. The model was simply reading the answer key. To fix this, I established an explicit drop list to purge all post-event and target columns before training. Stripping out the leaks dropped the model to an honest, realistic baseline.

### 3. Environment Pivoting: From Optuna to RandomizedSearchCV
When setting up hyperparameter optimization with Optuna, the local environment threw missing dependency errors. Instead of losing time troubleshooting package installations, I pivoted immediately to Scikit-learn's `RandomizedSearchCV`. Utilizing GPU acceleration allowed the pipeline to evaluate hundreds of parameter combinations quickly using installed dependencies.

### 4. Model Selection: Why XGBoost
Linear models assume straight-line relationships, but banking churn patterns are non-linear:

- **Credit score** risk shows minimal variance between average churned (645) and retained (652) accounts, yet risk spikes sharply in a left-tail cluster below 400.
- **Product holdings** follow a J-curve: churn drops to 7.60% at 2 products, then jumps to 82.71% at 3 products and 100% at 4 products.

XGBoost handles these threshold interactions natively by splitting on non-linear boundaries and outputting feature importances to validate logic against exploratory data analysis.

### 5. Threshold Optimization
A standard machine learning model uses a default 0.50 probability cutoff. However, in retail banking, missing a departing customer (false negative) is far more expensive than sending an unnecessary retention offer (false positive).

Evaluating the Precision-Recall curve demonstrated that a 0.50 cutoff dropped churn recall significantly, missing nearly half of all departing accounts. By optimizing for maximum F1-score, I lowered the decision threshold to **0.2909**.

| Metric | @0.50 Cutoff | @0.2909 Threshold |
|---|---|---|
| Churn Recall | ~46% | **69%** |
| Churn Precision | ~72% | **59%** |
| ROC-AUC Score | 0.8751 | **0.8751** |

Lowering the threshold to 0.2909 caught nearly 70% of actual churners while maintaining a 59% precision rate, providing an optimal balance for frontline retention teams.

---

## The 10 Questions the Business Actually Asked

Each diagnostic follows a clear structure: what happened, why it happened, and what to do next.

### 1. Who are our churned customers? (Demographic Profile)
**What happened:** The 41-60 age group represents **60.65% of total exits** (1,236 of 2,038 churned accounts). Female customers churned at 25.07% compared to 16.47% for male customers.

**Why it happened:** Middle-aged customers carry higher balances and complex product requirements, making them key targets for competitor bank promotions.

**What to do next:** Establish age-segmented retention outreach, specifically assigning dedicated relationship managers to high-balance clients in the 41-60 bracket.

### 2. Is churn linked to satisfaction scores or complaints?
**What happened:** Satisfaction scores showed zero predictive correlation (churn remained flat between 19.64% and 21.80%). However, 99.80% of churned customers had a logged complaint.

**Why it happened:** Complaints occur at the end of a dispute cycle, signaling that account closure is already imminent.

**What to do next:** Treat all logged complaints as immediate, high-priority intervention triggers rather than standard support tickets.

### 3. Do low engagement, inactivity, or lack of a credit card predict churn?
**What happened:** Inactive members churn at **26.87%** versus **14.27%** for active members. Credit card ownership showed no meaningful variance (20.81% vs. 20.20%).

**Why it happened:** Behavioral engagement directly impacts retention, while static product holding (like owning a credit card) does not guarantee active use.

**What to do next:** Trigger automated re-engagement workflows based on activity drop-offs rather than card issuance metrics.

### 4. How do balance, salary, tenure, and credit score affect loyalty?
**What happened:** Churn rates for medium and high balance tiers sit near 24%, compared to 14.25% for low balance accounts. Salary and tenure showed minimal correlation, while credit scores below 400 spiked churn to 32.28%.

**Why it happened:** High-balance clients possess higher mobility and capital to switch institutions. Extremely low credit scores signal financial stress.

**What to do next:** Prioritize retention budget allocation by balance tier rather than customer tenure.

### 5. Which card types drive churn, and does point accumulation help?
**What happened:** Churn rates across card types remained uniform (19.26% to 21.78%). Point accumulation tiers also showed flat churn rates (19.86% to 21.04%).

**Why it happened:** Current reward structures do not offer sufficient tier differentiation to incentivize account stickiness.

**What to do next:** Restructure the loyalty program so point accrual scales directly with account balance depth and relationship duration.

### 6. Can we build an early-detection risk model?
**What happened:** Yes. The trained XGBoost model achieved an **87.51% ROC-AUC score**. At the 0.2909 threshold, it catches **69% of churners** in advance with 59% precision.

**Why it happened:** Removing target proxy variables forced the model to learn genuine pre-churn behavioral patterns, such as balance-to-salary ratios and activity drop-offs.

**What to do next:** Export model prediction scores directly into a Power BI operational list for daily retention execution.

---

## Machine Learning: From Baseline to Deployable Model

**Feature Importance Summary:**

| Feature | Predictive Weight |
|---|---|
| Age | ~28% |
| Credit Score | ~24% |
| Balance / Balance-to-Salary Ratio | ~22% |
| Number of Products | ~14% |
| Active Status | ~6% |
| Credit Per Age | ~4% |
| Other Features | ~2% |

The top features account for the vast majority of the model's decisions, aligning directly with trends observed during exploratory analysis.

**Final Operating Metrics (@0.2909 Threshold):**
- **ROC-AUC:** 0.8751 (87.5%)
- **Churn Recall:** 69%
- **Churn Precision:** 59%

---

## Power BI: Turning a Model Into a Daily Habit

### Page 1 - Customer Overview
Establishes baseline portfolio metrics across 10,000 accounts, $764.86M in deposits, and credit tier distributions. Demonstrates that overall portfolio credit quality is high, emphasizing the need to preserve high-value accounts.
<img width="999" height="575" alt="image" src="https://github.com/user-attachments/assets/f8ddad82-23fb-49ab-adde-cce46b854f86" />

**Key Insights:**

- 🎁 **Loyalty Flaw:** Tenure has zero correlation with reward points accumulated, meaning long-term clients are not being incentivized to stay.
- 🌐 **Market Exposure:** France holds over 50% of total customers (5,000 accounts), making it the primary regional revenue driver.
- 💳 **Credit Stability:** Nearly 70% of clients maintain a "Good" credit rating, indicating low baseline portfolio risk.
- 💰 **Capital Anchor:** $764.86M in total balances across 10,000 accounts ($76.5K average balance per client).



### Page 2 - Historical Churn Analysis
Documents historical losses of 2,038 accounts ($185.68M capital lost, averaging $91.11K per departed account). Highlights regional concentrations in Germany (39.94% of churn) and France (39.79% of churn), establishing Germany as a high-churn-rate market relative to its customer base.
<img width="994" height="578" alt="image" src="https://github.com/user-attachments/assets/dbf22b57-d773-477d-87c8-f8f89171e72f" />

**Key Insights:**

- 💸 **Capital Drain:** $185.68M in total balance lost across 2,038 churned users, averaging a massive $91.11K per departed account.
- ⚠️ **Complaint Trigger:** 99.80% of departed clients filed an official complaint prior to leaving, making unresolved service issues the core churn driver.
- 🎯 **Age Vulnerability:** The 41–60 age bracket drives the highest risk volume by far, accounting for 1,236 total exits (over 60% of total churn).
- 🌐 **Regional Hotspots:** Germany (39.94%) and France (39.79%) suffer the vast majority of customer losses, making up nearly 80% of total churn.
- 📦 **Single-Product Risk:** Exit rates peak heavily among single-product holders, proving multi-product adoption is essential for account stickiness.

### Page 3 - Churn Prediction & Risk Pipeline
Translates model probability outputs into operational targets.
<img width="959" height="545" alt="image" src="https://github.com/user-attachments/assets/f147695e-5d76-49c8-9a4f-5fd4516ee777" />

**Key Insights:**

- **Capital at Risk:** **$73.75M** in balance vulnerable across **798 flagged accounts** (11.1% of portfolio) at the 0.2909 decision threshold.
- **Age Vulnerability:** The **41-60 age bracket** accounts for **72.8%** (581 of 798) of total churn risk, with inactive clients driving the majority of potential attrition (331 inactive accounts).
- **Operational Rescue List:** Filters high-risk accounts (sorted by individual probability up to 91%) concentrated in Germany and Spain among single-product holders for immediate outreach.

---

## Strategic Recommendations

1. **Deploy the 798-account rescue list immediately**, prioritizing the 581 middle-aged, inactive, high-credit accounts.
2. **Establish a Germany-specific retention campaign** to address structural market churn.
3. **Cross-sell single-product clients to two products** (where churn drops to 7.60%), while enforcing guardrails against pushing clients to 3+ products.
4. **Automate immediate escalation for logged complaints**, treating them as high-severity intervention events.
5. **Feed 0.2909 threshold model probability scores directly into CRM task queues**, ordered by individual risk score.

---

## Repository Structure
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
## How to Run This Project

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/bank-churn-prediction.git](https://github.com/yourusername/bank-churn-prediction.git)

**2. Install Python dependencies**

```bash
pip install -r requirements.txt

**3. Run the model pipeline**
Execute the notebooks in `notebooks/` in order:
* `01_data_cleaning_eda.ipynb` for data cleaning, feature engineering, and exploratory analysis.
* `02_model_training_evaluation.ipynb` for XGBoost training, leakage removal, `RandomizedSearchCV` hyperparameter tuning, and 0.2909 threshold optimization.

**4. Open the dashboard**
Open `reports/bank_churn_dashboard.pbix` in Power BI Desktop to view the interactive report, decomposition tree, and Operational Rescue List.

