# 🏦 Loan Approval Prediction using Decision Tree

This project implements a **classification model** to predict loan approval status based on applicant financial and demographic information using a **Decision Tree Classifier**. The workflow covers preprocessing, training, prediction, and automated creation of a submission file.

---

## 📁 Dataset

- **Files:** `train.csv`, `test.csv`
- **Features:**
  - `Gender`
  - `Married`
  - `Dependents`
  - `Education`
  - `Self_Employed`
  - `ApplicantIncome`
  - `CoapplicantIncome`
  - `LoanAmount`
  - `Loan_Amount_Term`
  - `Credit_History`
  - `Property_Area`
- **Target Column:**
  - `Loan_Status`: Y (Approved), N (Not Approved)

---

## 🔧 Technologies Used

- Python 🐍
- pandas
- scikit-learn (Decision Tree Classifier, LabelEncoder, preprocessing)

---

## 🚀 Steps Performed

1. Loaded training and test datasets (`train.csv`, `test.csv`)
2. Filled missing values with suitable defaults (mode/median)
3. Applied **Label Encoding** to categorical variables
4. Mapped target variable: `Loan_Status` (Y→1, N→0)
5. Selected relevant features for modeling
6. Trained a **Decision Tree Classifier** (`max_depth=4`) on the training data
7. Made predictions on the test set
8. Translated predicted labels to `Y`/`N`
9. Generated a `loan_submission.csv` with predictions

---

## 📌 How to Run

1. Download or clone this repository.
2. Install required Python libraries (`pandas`, `scikit-learn`).
3. Ensure `train.csv` and `test.csv` are in the same directory as the script.
4. Run the script in terminal: python loan_predictor.py

