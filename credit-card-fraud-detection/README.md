# 💳 Credit Card Fraud Detection using Logistic Regression

This project is a basic implementation of a **classification model** to detect fraudulent credit card transactions using **Logistic Regression**. The dataset is highly imbalanced, and our goal is to understand how to preprocess, train, and predict fraud probabilities.

---

## 📁 Dataset

- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Features:** 
  - `V1` to `V28`: PCA-transformed features (for anonymity)
  - `Amount`: Transaction amount (scaled)
  - `Time`: Time elapsed since the first transaction (dropped)
- **Target column:**
  - `Class`: 0 for legitimate, 1 for fraud

---

## 🔧 Technologies Used

- Python 🐍
- pandas
- scikit-learn (Logistic Regression, preprocessing, train-test split)
- StandardScaler

---

## 🚀 Steps Performed

1. Loaded the dataset
2. Dropped irrelevant `Time` column
3. Scaled `Amount` feature using `StandardScaler`
4. Split data into training and test sets
5. Trained Logistic Regression model
6. Predicted and printed the **probabilities of fraud** for test samples

---

## 📌 How to Run

1. Clone this repo or download the `.py` file and dataset.
2. Place `creditcard.csv` in the same directory.
3. Run the script:


