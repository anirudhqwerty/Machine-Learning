# Hierarchical Clustering on Mall Customers

This project performs **customer segmentation** using **Hierarchical Clustering** on the popular **Mall Customers dataset**. The goal is to group similar customers based on their **Annual Income** and **Spending Score**.

---

## 📁 Dataset

- **Source:** [Mall_Customers.csv](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
- **Features Used:**
  - `Annual Income (k$)`
  - `Spending Score (1-100)`

---

## 🔍 Objective

To cluster mall customers into distinct groups using **Agglomerative Clustering** based on spending behavior and income level.

---

## 🧪 Steps Performed

1. **Data Loading**
2. **Feature Selection**
3. **Standardization** using `StandardScaler`
4. **Agglomerative Clustering** with:
   - `n_clusters = 5`
   - `linkage = 'ward'`
   - (No need to specify `affinity` with `'ward'` linkage in latest scikit-learn)
5. **Saving Results** with cluster labels

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn

---

## 📦 Output

- Final CSV: `mall_hierarchical_clusters.csv` containing original data + predicted cluster labels.

---

## 🚀 How to Run

1. Download `Mall_Customers.csv` from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial)
2. Place it in the same directory as the Python script.
3. Run the script:
   ```bash
   python hierarchical_clustering_mall.py
