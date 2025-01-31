import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Load the preprocessed (already standardized) dataset
processed_file_path = "C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 3/processed_air_quality.csv"
df_processed = pd.read_csv(processed_file_path)

# Step 1: Perform hierarchical clustering using Ward's method
Z = linkage(df_processed, method='ward')

# Step 2: Plot the Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(Z)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig("dendrogram.png", dpi=300)
plt.clf()

# Step 3: Assign cluster labels based on a distance threshold
max_d = 3  # Maximum distance for forming clusters (threshold for cutting the dendrogram)
clusters = fcluster(Z, max_d, criterion='distance')

# Step 4: Add cluster labels to the dataframe
df_processed['Cluster'] = clusters

# Step 5: Print cluster sizes
print("\nCluster Sizes:")
cluster_sizes = df_processed['Cluster'].value_counts()
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: {size} data points")

# Step 6: Calculate and print centroids for each cluster (mean values of features)
print("\nCluster Centroids (Mean Values):")
centroids_df = df_processed.groupby('Cluster').mean()
for cluster, centroid in centroids_df.iterrows():
    print(f"\nCluster {cluster} Centroid:")
    for feature, value in centroid.items():
        print(f"  {feature}: {value:.4f}")
