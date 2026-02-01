## Disease Prediction from Medical Data

### Objective
To predict the presence of heart disease using clinical patient data.

### Dataset
UCI Heart Disease Dataset (Features: age, sex, cholesterol, chest pain type, max heart rate, etc.)

## Tech Stack
* **Language:** Python 3.12
* **Libraries:** `pandas`, `scikit-learn`, `imbalanced-learn`, `xgboost`
* **Dataset:** heart_cleveland_upload.csv (via Kaggle)

### Feature Engineering
The dataset looks clean and are already preprocessed to numerical values

### Implementation Details
* **Algorithms Compared:** Logistic Regression, Random Forest, SVM, and XGBoost.
* **Preprocessing:** Applied `StandardScaler` for SVM/Logistic Regression to ensure distance-based feature parity.
* **Optimization:** Used XGBoost for gradient boosting performance and Random Forest for ensemble stability.

### Key Results
* **Best Performing Model:** Final Recommendation: SVM
* **Clinical Insight:** Using SHAP analysis, "Chest Pain Type" and "Max Heart Rate" were identified as the strongest predictors of heart disease risk.

---

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the script: `python main.py`.