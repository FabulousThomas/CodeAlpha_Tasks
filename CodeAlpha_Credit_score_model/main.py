import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Machine Learning Imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# 1. LOAD DATA FROM KAGGLE
# file_path = "averkiyoliabev/home-equity-line-of-creditheloc"
# df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, file_path, 'heloc_dataset_v1 (1).csv')

# LOAD CSV DATA
df = pd.read_csv('data/heloc_dataset_v1.csv')

print(df.head())

# 2. FEATURE ENGINEERING & CLEANING
df['RiskPerformance'] = df['RiskPerformance'].map({'Good': 0, 'Bad': 1})
df = df[df['ExternalRiskEstimate'] != -9]  # Drop rows with no credit file
df.replace(-7, 999, inplace=True)         # -7 means "no delinquency" -> high value

for col in df.columns:
    if col != 'RiskPerformance':
        median_val = df[df[col] > 0][col].median()
        df[col] = df[col].replace(-8, median_val)
        
        
# 3. SPLIT DATA
X = df.drop(columns=['RiskPerformance'])
y = df['RiskPerformance']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=98, test_size=0.2)


# 4. MULTI-MODEL COMPARISON (The Baseline)
# Note: We use an imblearn Pipeline to scale data for Logistic Regression correctly
models = {
    'Logistic Regression': ImbPipeline([('scaler', StandardScaler()), 
                                       ('clf', LogisticRegression(class_weight='balanced', max_iter=2000))]),
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced', max_depth=7, random_state=42),
    'Random Forest': RandomForestClassifier(class_weight='balanced', n_estimators=100, max_depth=7, random_state=42)
}

print("--- Initial Model Comparison ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"{name} AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("-" * 50)
    
    
# 5. HYPERPARAMETER TUNING (Boosting the Winner)
print("\n--- Starting GridSearch for Random Forest ---")
param_grid = {
    'clf__n_estimators': [200, 500],
    'clf__max_depth': [10, 15, None],
    'clf__min_samples_split': [2, 10]
}

# Pipeline with SMOTE for the final tuning
tuning_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

grid_search = GridSearchCV(tuning_pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"Best Params: {grid_search.best_params_}")
print(f"Tuned AUC: {roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1]):.4f}")


# 6. MODEL INTERPRETABILITY WITH SHAP
print("\n--- Generating SHAP Explanations ---")
# Use the classifier part of the pipeline for SHAP
explainer = shap.TreeExplainer(best_model.named_steps['clf'])
# Transform X_test through the pipeline steps before the classifier
X_test_preprocessed = best_model.named_steps['scaler'].transform(X_test)
shap_values = explainer.shap_values(X_test_preprocessed)

# Plot Summary
shap.summary_plot(shap_values[:, :, 1], X_test_preprocessed, feature_names=X.columns)

print("Completed!")