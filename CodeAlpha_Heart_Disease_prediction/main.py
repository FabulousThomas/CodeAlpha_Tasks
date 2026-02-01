import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.pipeline import Pipeline as ImbPipeline
import kagglehub

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=98)

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
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=98)
}

# 3. EVALUATE MODELS
print("--- Medical Disease Prediction Comparison ---")
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = auc
    
    print(f"{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC:  {auc:.4f}")
    print("-" * 30)

# 4. IDENTIFY BEST MODEL
best_model_name = max(results, key=results.get)
print(f"Final Recommendation: {best_model_name}")
print("Completed!")