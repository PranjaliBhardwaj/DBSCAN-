import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

filename = "CSV_BIG.csv"
path_dir = ".\\"

# Read the CSV file
dataframe = pd.read_csv(path_dir + filename, encoding="utf-8", sep=';')
df = dataframe.copy(deep=True)

# Assuming all columns are used for clustering
features = df.values

# Number of neighbors to consider
k = 5  # Often, min_samples + 1 is used, here min_samples is set to 4, so k=5

# Fit the NearestNeighbors model
nearest_neighbors = NearestNeighbors(n_neighbors=k)
nearest_neighbors.fit(features)

# Compute the distances to the k-nearest neighbors
distances, indices = nearest_neighbors.kneighbors(features)

# Sort the distances to the k-th nearest neighbor
distances = np.sort(distances[:, k-1], axis=0)

# Plot the k-distance graph
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.axhline(y=distances.mean(), color='r', linestyle='--')
plt.title('k-NN Distance Plot')
plt.xlabel('Points sorted by distance to {}-th nearest neighbor'.format(k))
plt.ylabel('{}-th nearest neighbor distance'.format(k))
plt.grid(True)
plt.show()
