# 📧 Email Spam Detection using Logistic Regression

This project implements a **binary classification model** to detect whether an email is **spam** or **not spam** using **Logistic Regression** and **TF-IDF Vectorization**. The goal is to build a lightweight yet effective machine learning pipeline for basic text classification tasks.

---

## 📁 Dataset

- **Source:** [SMS Spam Collection Dataset (UCI)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Download Link Used:** [GitHub CSV](https://raw.githubusercontent.com/tae898/Spam-Detection/master/spam.csv)
- **Features:**
  - `text`: Raw message content (SMS/email style)
- **Target Column:**
  - `label`: `'spam'` or `'ham'` (mapped to 1 or 0)

---

## 🔧 Technologies Used

- Python 🐍
- pandas
- scikit-learn
  - `TfidfVectorizer` for feature extraction
  - `LogisticRegression` for model training
  - `train_test_split` for dataset splitting
  - Evaluation metrics (Accuracy, Confusion Matrix, F1-score)

---

## 🚀 Steps Performed

1. Loaded the spam dataset from a GitHub link
2. Selected and renamed relevant columns (`v1` → `label`, `v2` → `text`)
3. Mapped categorical labels (`ham` → 0, `spam` → 1)
4. Vectorized the text using **TF-IDF** to convert it into numerical form
5. Split the data into training and test sets
6. Trained a **Logistic Regression** model
7. Evaluated the model using Accuracy, Confusion Matrix, and Classification Report
8. Made predictions on custom sample messages to test spam detection

---

## 📌 How to Run

1. Clone this repo or download the `.py` file
2. Make sure you’re connected to the internet (for GitHub CSV loading)
3. Run the script:python spam_detector.py
