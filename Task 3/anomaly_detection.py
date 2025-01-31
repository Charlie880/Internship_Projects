import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# Load the preprocessed (already standardized) dataset
processed_file_path = "C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 3/processed_air_quality.csv"
df_processed = pd.read_csv(processed_file_path)

# Step 1: Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_processed['Cluster'] = kmeans.fit_predict(df_processed)

# Step 2: Calculate the Euclidean distance from each data point to its assigned cluster centroid
distances = cdist(df_processed.iloc[:, :-1], kmeans.cluster_centers_, 'euclidean')

# Get the distance of each point from its assigned centroid (based on its cluster)
df_processed['Distance_from_Centroid'] = distances[np.arange(distances.shape[0]), df_processed['Cluster']]

# Step 3: Set a threshold for outliers (e.g., distance above the 95th percentile of all distances)
threshold = np.percentile(df_processed['Distance_from_Centroid'], 95)

# Identify anomalies (outliers) that exceed the threshold
outliers = df_processed[df_processed['Distance_from_Centroid'] > threshold]

# Step 4: Apply PCA to reduce data to 2D for visualization
pca = PCA(n_components=2)
pca_components = pca.fit_transform(df_processed.iloc[:, :-2])

# Step 5: Visualize the anomalies on the PCA-transformed plot
plt.figure(figsize=(10, 6))
plt.scatter(pca_components[:, 0], pca_components[:, 1], c=df_processed['Cluster'], cmap='viridis', label='Data Points')
plt.scatter(pca_components[outliers.index, 0], pca_components[outliers.index, 1], color='red', label='Anomalies', marker='x')

plt.title("PCA Visualization with Anomalies")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("Anomalies")

# Step 6: Output the outliers and their details
print(f"\nIdentified Anomalies (Outliers):")
print(outliers[['Cluster', 'Distance_from_Centroid']])

# Optionally, print the number of outliers and the threshold
print(f"\nNumber of Outliers Detected: {outliers.shape[0]}")
print(f"Outlier Threshold (95th Percentile Distance): {threshold}")
