# Creditworthiness Prediction: FICO HELOC Analysis

## 📌 Project Objective
This project builds a robust machine learning pipeline to predict an individual's creditworthiness using the **FICO HELOC (Home Equity Line of Credit)** dataset. The goal is to classify applicants as "Good" or "Bad" credit risks while maintaining high **explainability**—a critical requirement in regulated financial industries.

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn` (SMOTE), `SHAP` (Explainability), `matplotlib/seaborn`
* **Dataset:** FICO HELOC (via Kaggle)

---

## 🏗️ The "Architectural" Approach

### 1. Advanced Data Engineering
I implemented custom handling for FICO-specific "Special Values":
* **-9 (No Bureau Record):** Rows dropped as they lack predictive signals.
* **-7 (Condition Not Met):** Replaced with $999$ to represent "infinite" time since a negative event (low risk).
* **-8 (Missing/Invalid):** Imputed with the feature median to maintain volume.

### 2. The Integrated ML Pipeline
To prevent **Data Leakage**, I utilized the `imblearn` Pipeline:
* **Standardization:** `StandardScaler` for Logistic Regression convergence.
* **Class Balancing:** **SMOTE** integrated within cross-validation to handle imbalance.
* **Feature Selection:** **Recursive Feature Elimination (RFE)** to identify the top 12 predictive variables.

### 3. Hyperparameter Optimization
Performed `GridSearchCV` on the **Random Forest Classifier**, optimizing for **ROC-AUC** to ensure effective risk discrimination.

---

## 🔍 Explainability (XAI)
Using **SHAP**, I decomposed the model's "Black Box" into actionable insights:
* **Top Risk Driver:** `ExternalRiskEstimate`. Higher scores drastically reduce default probability.
* **Credit Utilization:** `NetFractionRevolvingBurden` is a primary predictor of high-risk behavior.

"The final Tuned Random Forest model achieved an AUC of 0.805, outperforming the baseline Decision Tree by a significant margin. By optimizing for max_depth and min_samples_split, the model achieved a high Recall (0.75) for high-risk applicants, successfully balancing the trade-off between identifying defaulters and maintaining overall accuracy. The close performance of the Logistic Regression model suggests that the feature engineering successfully captured the linear risk signals inherent in the FICO dataset."

---

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the script: `python main.py`.

---

## 📜 License
Distributed under the MIT License.