import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import lightgbm as lgb

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(df.head())
print(df['target'].value_counts())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_preds)
print("XGBoost Accuracy:", xgb_acc)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)
lgb_preds = lgb_model.predict(X_test)
lgb_acc = accuracy_score(y_test, lgb_preds)
print("LightGBM Accuracy:", lgb_acc)

print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_preds))

print("LightGBM Classification Report:")
print(classification_report(y_test, lgb_preds))

print("XGBoost Confusion Matrix:")
print(confusion_matrix(y_test, xgb_preds))

print("LightGBM Confusion Matrix:")
print(confusion_matrix(y_test, lgb_preds))
