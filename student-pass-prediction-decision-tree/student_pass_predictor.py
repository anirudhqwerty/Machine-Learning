import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

df = pd.read_csv("student-mat.csv", sep=';')
df['pass'] = df['G3'] >= 10
df['pass'] = df['pass'].map({True: 1, False: 0})
df = df.drop(['G1', 'G2', 'G3'], axis=1)

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

X = df.drop('pass', axis=1)
y = df['pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_gini = DecisionTreeClassifier(criterion='gini', max_depth=4)
tree_gini.fit(X_train, y_train)

tree_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=4)
tree_entropy.fit(X_train, y_train)

pred_gini = tree_gini.predict(X_test)
pred_entropy = tree_entropy.predict(X_test)

print("Gini Accuracy:", accuracy_score(y_test, pred_gini))
print("Entropy Accuracy:", accuracy_score(y_test, pred_entropy))

plt.figure(figsize=(20,10))
plot_tree(tree_gini, feature_names=X.columns, class_names=["Fail", "Pass"], filled=True)
plt.title("Decision Tree (Gini)")
plt.show()
