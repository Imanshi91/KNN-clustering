import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('Mall_Customers.csv')

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_normalized)
    inertias.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Based on the elbow curve, let's choose k=5
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_normalized)

# Add cluster labels to the original dataframe
data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.figure(figsize=(10, 8))
for i in range(k):
    cluster = data[data['Cluster'] == i]
    plt.scatter(cluster['Annual Income (k$)'], cluster['Spending Score (1-100)'], label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0] * scaler.scale_[0] + scaler.mean_[0],
            kmeans.cluster_centers_[:, 1] * scaler.scale_[1] + scaler.mean_[1],
            s=300, c='yellow', label='Centroids')

plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segments')
plt.legend()
plt.show()

# Print summary statistics for each cluster
for i in range(k):
    cluster = data[data['Cluster'] == i]
    print(f"\nCluster {i} Summary:")
    print(cluster[['Annual Income (k$)', 'Spending Score (1-100)']].describe())