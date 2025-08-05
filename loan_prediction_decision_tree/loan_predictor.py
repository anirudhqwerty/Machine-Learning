import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.head())

for df in [train, test]:
    df['Gender'].fillna('Male', inplace=True)
    df['Married'].fillna('Yes', inplace=True)
    df['Dependents'].fillna('0', inplace=True)
    df['Self_Employed'].fillna('No', inplace=True)
    df['Credit_History'].fillna(1.0, inplace=True)
    df['Loan_Amount_Term'].fillna(360.0, inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
# will use a model to predict the missing values in the later version



le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])

train['Loan_Status'] = train['Loan_Status'].map({'Y': 1, 'N': 0})

features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
            'Credit_History', 'Property_Area']

X = train[features]
y = train['Loan_Status']

model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)

X_test = test[features]
predictions = model.predict(X_test)

predicted_labels = ['Y' if i == 1 else 'N' for i in predictions]

submission = pd.DataFrame({
    'Loan_ID': test['Loan_ID'],
    'Loan_Status': predicted_labels
})

submission.to_csv('loan_submission.csv', index=False)
print("Submission file created: loan_submission.csv")
