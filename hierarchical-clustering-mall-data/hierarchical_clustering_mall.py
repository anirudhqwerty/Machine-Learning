import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

model = AgglomerativeClustering(
    n_clusters=5,
    metric='euclidean',
    linkage='ward'
)
df['Cluster'] = model.fit_predict(scaled_data)

df.to_csv("mall_hierarchical_clusters.csv", index=False)
print("Cluster assignments saved to mall_hierarchical_clusters.csv")
