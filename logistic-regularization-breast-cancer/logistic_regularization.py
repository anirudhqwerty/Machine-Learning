from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model_l1.fit(X_train, y_train)
y_pred_l1 = model_l1.predict(X_test)
acc_l1 = accuracy_score(y_test, y_pred_l1)

model_l2 = LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, max_iter=1000)
model_l2.fit(X_train, y_train)
y_pred_l2 = model_l2.predict(X_test)
acc_l2 = accuracy_score(y_test, y_pred_l2)

print("🔍 Accuracy Comparison:")
print(f"L1 Regularization (Lasso): {acc_l1:.4f}")
print(f"L2 Regularization (Ridge): {acc_l2:.4f}")
