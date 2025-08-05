# 🎓 Student Performance Prediction using Decision Trees

This project predicts whether a student will pass or fail based on their attributes using a **Decision Tree Classifier**. The model compares Gini and Entropy criteria, visualizes the decision tree, and demonstrates end-to-end data preprocessing and evaluation.

---

## 📁 Dataset

- **File:** `student-mat.csv`
- **Source:** UCI Machine Learning Repository - Student Performance Dataset
- **Features:** Various academic and demographic attributes (e.g., school, sex, age, address, studytime, absences, and more)
- **Target Column:**  
  - `pass`: 1 (Passed, G3 ≥ 10), 0 (Failed, G3 < 10)

---

## 🔧 Technologies Used

- Python 🐍
- pandas
- scikit-learn (LabelEncoder, train_test_split, DecisionTreeClassifier, plot_tree, accuracy_score)
- matplotlib (for visualization)

---

## 🚀 Steps Performed

1. Loaded data from `student-mat.csv` (semicolon-separated values)
2. Created a new column `pass` based on final grade (`G3`), then dropped `G1`, `G2`, and `G3`
3. Encoded all categorical features using **Label Encoding**
4. Split data into training and test sets (80/20 split, random_state=42)
5. Trained two Decision Tree Classifiers:
   - One with the **Gini** criterion
   - One with the **Entropy** criterion
6. Evaluated both models on the test set using **accuracy**
7. Visualized the decision tree (Gini) using `matplotlib`

---

## 📌 How to Run

1. Download or clone this repository.
2. Ensure Python and the required libraries are installed (`pandas`, `scikit-learn`, `matplotlib`).
3. Place `student-mat.csv` in the script directory.
4. Run the script in terminal: python student-mat.csv
