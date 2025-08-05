# 🎗️ Breast Cancer Prediction using Logistic Regression

This project is a basic implementation of a **classification model** to predict whether a tumor is **benign** or **malignant** using **Logistic Regression** on the Breast Cancer Wisconsin dataset. The goal is to preprocess, train, evaluate, and make predictions using real-world health features.

---

## 📁 Dataset

- **Source:** Built-in dataset from `sklearn.datasets`
- **Features:**  
  - Includes 30 numeric features such as:
    - `mean radius`, `mean texture`, `mean smoothness`, etc.
- **Target Column:**
  - `0`: Malignant  
  - `1`: Benign

---

## 🔧 Technologies Used

- Python 🐍  
- scikit-learn (Logistic Regression, data loading, splitting, evaluation)
- pandas & numpy

---

## 🚀 Steps Performed

1. Loaded the Breast Cancer dataset from `sklearn`
2. Split into features (`X`) and target (`y`)
3. Performed a train-test split (80/20)
4. Trained a Logistic Regression model
5. Evaluated using accuracy, confusion matrix, and classification report
6. Took **real-time user input** to predict tumor type

---

## 📌 How to Run

1. Download or clone this repository.
2. Make sure Python and required libraries are installed.
3. Run the script in terminal:  
   ```bash
   python breast_cancer_checker.py
