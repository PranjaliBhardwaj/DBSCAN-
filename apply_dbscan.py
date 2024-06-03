# Once you have determined an appropriate value for eps from the k-distance plot, you can use it to apply DBSCAN:
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Choose the eps value based on the k-distance plot
eps_value = 0.5  # Example value, replace with the value determined from the plot

dbscan = DBSCAN(eps=eps_value, min_samples=5)
cluster_labels = dbscan.fit_predict(features)

if len(set(cluster_labels)) > 1:
    silhouette_avg = silhouette_score(features, cluster_labels)
    print(f"For eps = {eps_value}, the average silhouette_score is: {silhouette_avg}")
else:
    print(f"For eps = {eps_value}, no valid silhouette score (only one cluster or all points are noise).")
