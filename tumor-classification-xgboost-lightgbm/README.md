# 🧠 Tumor Classification using XGBoost and LightGBM

This project implements two powerful **machine learning classifiers**—**XGBoost** and **LightGBM**—to predict whether a tumor is **malignant** (dangerous) or **benign** (harmless). It uses the built-in **Breast Cancer dataset** from `sklearn.datasets`.

---

## 📁 Dataset

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features:**
  - 30 numerical features related to tumor measurements (e.g., mean radius, mean texture, etc.)
- **Target column:**
  - `target`: `0` for malignant, `1` for benign

---

## 🔧 Technologies Used

- Python 🐍
- pandas
- scikit-learn (model selection, metrics, datasets)
- XGBoost
- LightGBM

---

## 🚀 Steps Performed

1. Loaded the Breast Cancer dataset using `sklearn`
2. Converted it into a `pandas` DataFrame
3. Split data into training and testing sets
4. Trained both **XGBoost** and **LightGBM** models
5. Made predictions on the test set
6. Evaluated models using **accuracy**, **confusion matrix**, and **classification report**
7. Compared their performance

---

## 📌 How to Run

1. Clone this repo or download the `.py` file.
2. Make sure you have the required libraries installed:
        pip install pandas scikit-learn xgboost lightgbm
3. Run the script: python tumor_classifier.py