# Credit Risk Simulation & Expected Loss Analytics

## Executive Overview
This project builds an end-to-end Machine Learning pipeline to assess retail credit risk, predict **Probability of Default (PD)**, assign behavioral credit scores (300–850), and estimate total **Expected Loss (EL)** in GHS across portfolio risk tiers.

##Dataset link: https://drive.google.com/drive/folders/1dnKFeJYxUjwSVgf8HFbJR1o34vj27YXC?usp=drive_link

## Portfolio Risk Summary

| Risk Tier | Risk Level | Score Range | Avg PD | Exposure (GHS) | Expected Loss (GHS) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1** | Prime | 750–850 | ~9.5% | GHS 1,001,713 | GHS 53,658 |
| **Tier 2** | Standard | 670–749 | ~24.9% | GHS 705,122 | GHS 96,881 |
| **Tier 3** | Subprime | 580–669 | ~40.6% | GHS 567,112 | GHS 128,401 |
| **Tier 4** | Critical | < 580 | ~65.4% | GHS 997,311 | GHS 375,894 |

## Project Architecture
- **Data Ingestion & Cleaning:** Standardizing raw applicant attributes (`Credit-Risk-Dataset.csv`).
- **Feature Pipeline:** `scikit-learn` `ColumnTransformer` with `StandardScaler` and `OneHotEncoder`.
- **Model Training:** Logistic Regression baseline vs. hyperparameter-tuned `GradientBoostingClassifier`.
- **Risk Metrics Engine:** Calculating Expected Loss via:
  $$EL = PD \times EAD \times LGD$$
  *(where $LGD = 0.55$)*

##  How to Run
1. Clone the repository: 
   ```bash
   git clone [https://github.com/kpankpa/credit-risk-simulation.git](https://github.com/kpankpa/credit-risk-simulation.git)
   cd credit-risk-simulation# Credit-risk-simulation
