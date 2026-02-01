import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.pipeline import Pipeline as ImbPipeline
import kagglehub
import matplotlib.pyplot as plt
import numpy as np

# 1. LOAD MEDICAL DATA
# Download latest version
path = kagglehub.dataset_download("cherngs/heart-disease-cleveland-uci")
df_med = pd.read_csv(path + '/heart_cleveland_upload.csv')

print(df_med.head())
print('--' * 20)
print('DATASET INFO')
print(df_med.info())
print('--' * 20)

X = df_med.drop(columns=['condition'])
y = df_med['condition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=98, stratify=y)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# 2. DEFINE ALL MODELS
# Note: SVM and Logistic Regression require scaling (StandardScaler)
models = {
    'Logistic Regression': ImbPipeline([
        ('scaler', StandardScaler()), 
        ('clf', LogisticRegression(random_state=98))
    ]),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=98),
    'SVM': ImbPipeline([
        ('scaler', StandardScaler()), 
        ('clf', SVC(probability=True, kernel='linear', random_state=98))
    ]),
    'XGBoost': XGBClassifier(
        eval_metric='logloss', 
        random_state=98, 
        scale_pos_weight=pos_weight
    )
}

# 3. EVALUATE MODELS
print("--- Medical Disease Prediction Comparison ---")
results = {}
roc_curves = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = auc
    
    # Store ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_curves[name] = (fpr, tpr, auc)
    
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC:  {auc:.4f}")
    print("-" * 30)

# 4. IDENTIFY BEST MODEL
best_model_name = max(results, key=results.get)
print(f"Final Recommendation: {best_model_name}")

# 5. VISUALIZATIONS
# 5.1 ROC Curve Plot
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange']
for i, (name, (fpr, tpr, auc)) in enumerate(roc_curves.items()):
    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
             label=f'{name} (AUC = {auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Heart Disease Prediction Models')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('images/roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 5.2 Feature Importance Analysis
print("\n--- Feature Importance Analysis ---")

# Show feature importance for Random Forest (even if not the best model)
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
feature_names = X.columns

# Sort features by importance
indices = np.argsort(importances)[::-1]

print(f"\nTop 10 Most Important Features (Random Forest):")
for i in range(min(10, len(feature_names))):
    print(f"{i+1:2d}. {feature_names[indices[i]]:15s}: {importances[indices[i]]:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 8))
plt.title('Feature Importance - Random Forest', fontsize=16)
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
# plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nNote: Best model is {best_model_name} with ROC-AUC: {results[best_model_name]:.4f}")
print("Feature importance shown for Random Forest for interpretability.")

print("Completed!")