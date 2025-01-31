import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the preprocessed dataset
processed_file_path = "C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 3/processed_air_quality.csv"
df_processed = pd.read_csv(processed_file_path)

# Determine the optimal number of clusters using the Elbow Method
inertia_values = []
K_range = range(1, 11)  # Checking for 1 to 10 clusters

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_processed)
    inertia_values.append(kmeans.inertia_)  # Inertia measures clustering performance

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia_values, marker='o', linestyle='--', color='b')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal K")
plt.tight_layout()
plt.savefig("elbow_method.png", dpi=300)
plt.close()

# Select the optimal K from the elbow point
optimal_k = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_processed['Cluster'] = kmeans.fit_predict(df_processed)


# --- Print Numerical Insights ---
print(f"\n K-Means clustering completed with {optimal_k} clusters!\n")
print("Cluster Sizes:")
print(df_processed['Cluster'].value_counts().sort_index())

print("\n Cluster Centroids (Standardized Values):")
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns=df_processed.columns[:-1])
print(centroids_df)

print(f"\n Final Inertia Value: {kmeans.inertia_:.2f}\n")

# Visualize the clusters (for 2 selected features)
plt.figure(figsize=(8, 6))
plt.scatter(df_processed.iloc[:, 0], df_processed.iloc[:, 1], c=df_processed['Cluster'], cmap='viridis', alpha=0.6)
plt.xlabel(df_processed.columns[0])
plt.ylabel(df_processed.columns[1])
plt.title("K-Means Clustering Visualization")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig("kmeans_clusters.png", dpi=300)
plt.close()