import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("https://raw.githubusercontent.com/tae898/Spam-Detection/master/spam.csv")

df = df[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

sample = ["Congratulations! You've won a free iPhone!",
          "Can we reschedule the meeting to tomorrow?"]

sample_vec = vectorizer.transform(sample)
predictions = model.predict(sample_vec)

for msg, label in zip(sample, predictions):
    print(f"Message: '{msg}' → {'Spam' if label == 1 else 'Not Spam'}")
