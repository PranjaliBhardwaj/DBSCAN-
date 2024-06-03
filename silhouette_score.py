import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

filename = "customer_data.csv"
path_dir = ".\\"

# Read the CSV file
dataframe = pd.read_csv(path_dir + filename, encoding="utf-8", sep=';')
df = dataframe.copy(deep=True)

# Assuming all columns are used for clustering
features = df.values

# Define a range of epsilon values to test
eps_values = np.arange(0.1, 5.0, 0.1)
print("Epsilon values from 0.1 to 5.0 with step 0.1:\n", eps_values)

for eps in eps_values:
    clusterer = DBSCAN(eps=eps, min_samples=5)  # Adjust min_samples if needed
    cluster_labels = clusterer.fit_predict(features)
    
    # Ignore silhouette score calculation if only one cluster or all points are noise
    if len(set(cluster_labels)) > 1 and -1 in cluster_labels and len(set(cluster_labels)) > 2:
        silhouette_avg = silhouette_score(features, cluster_labels)
        print(f"For eps = {eps}, the average silhouette_score is: {silhouette_avg}")
    elif len(set(cluster_labels)) > 1:
        silhouette_avg = silhouette_score(features, cluster_labels)
        print(f"For eps = {eps}, the average silhouette_score is: {silhouette_avg}")
    else:
        print(f"For eps = {eps}, no valid silhouette score (only one cluster or all points are noise).")
