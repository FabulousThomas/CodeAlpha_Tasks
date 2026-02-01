# 🩺 Heart Disease Prediction from Medical Data

## 📌 Objective

Build a machine learning model to predict the presence of heart disease
using clinical patient data. The project compares multiple
classification algorithms and evaluates them using clinically relevant
metrics.

------------------------------------------------------------------------

## 📊 Dataset

-   **Source:** UCI Heart Disease Dataset (Cleveland)\
-   **Accessed via:** Kaggle (`heart_cleveland_upload.csv`)\
-   **Features:** Age, sex, chest pain type, cholesterol, resting blood
    pressure, max heart rate, and more\
-   **Target:** Presence of heart disease (binary classification)

------------------------------------------------------------------------

## 🧰 Tech Stack

-   **Language:** Python 3.12\
-   **Libraries:**
    -   pandas\
    -   scikit-learn\
    -   imbalanced-learn\
    -   xgboost\
    -   matplotlib, numpy

------------------------------------------------------------------------

## 🧪 Feature Engineering

The dataset is already numerically encoded and relatively clean.\
Minimal preprocessing was required beyond: - Train/test split with
**stratification**\
- Scaling numerical features for distance-based models

------------------------------------------------------------------------

## ⚙️ Implementation Details

**Algorithms compared:** - Logistic Regression\
- Random Forest\
- Support Vector Machine (SVM)\
- XGBoost

**Preprocessing:** - Applied **StandardScaler** for Logistic Regression
and SVM\
- Tree-based models used raw feature values

**Evaluation Metrics:** - Accuracy\
- ROC-AUC\
- ROC Curves\
- Feature importance (Random Forest)

------------------------------------------------------------------------

## 📈 Key Results

  Model                 Accuracy     ROC-AUC
  --------------------- ------------ ------------
  Logistic Regression   0.8167       **0.8984**
  Random Forest         **0.8500**   0.8973
  SVM                   0.8000       0.8895
  XGBoost               0.8167       0.8772

➡ **Final Recommendation:** Logistic Regression (best ROC-AUC and
interpretability for clinical use)

------------------------------------------------------------------------

## 🔍 Clinical Insights

Feature importance analysis (using Random Forest) highlighted the most
influential predictors of heart disease risk:

-   **Thalassemia (thal)**\
-   **Max heart rate achieved (thalach)**\
-   **ST depression (oldpeak)**\
-   **Chest pain type (cp)**\
-   **Age and cholesterol**

These features align with known clinical risk factors, indicating that
the model learned meaningful medical patterns.

------------------------------------------------------------------------

## 🚀 How to Run

``` bash
pip install -r requirements.txt
python main.py
```

------------------------------------------------------------------------

## 🏁 Conclusion

Logistic Regression achieved the highest ROC-AUC (0.898), making it the
preferred model for this healthcare task due to its strong performance
and interpretability. The project demonstrates how multiple ML models
can be compared fairly for medical risk prediction while maintaining
transparency.
