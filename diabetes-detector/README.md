# 🩺 Diabetes Prediction using Logistic Regression

This project is a simple implementation of a **classification model** to predict whether a person has diabetes based on health-related features using **Logistic Regression**. The goal is to preprocess, train, evaluate, and allow user-based predictions through terminal input.

---

## 📁 Dataset

- **Source:** [Pima Indians Diabetes Dataset (GitHub)](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv)
- **Features:**
  - `Pregnancies`
  - `Glucose`
  - `BloodPressure`
  - `SkinThickness`
  - `Insulin`
  - `BMI`
  - `DiabetesPedigreeFunction`
  - `Age`
- **Target Column:**
  - `Outcome`: 0 (Non-diabetic), 1 (Diabetic)

---

## 🔧 Technologies Used

- Python 🐍
- pandas
- scikit-learn (Logistic Regression, preprocessing, train-test split, evaluation)

---

## 🚀 Steps Performed

1. Loaded dataset from GitHub
2. Split into features and target variable
3. Trained a Logistic Regression model
4. Evaluated with accuracy, confusion matrix, and classification report
5. Took real-time **user input** and predicted diabetes status

---

## 📌 How to Run

1. Download or clone this repository.
2. Ensure Python and required libraries are installed.
3. Run the script in terminal: python diabetes_checker.py

