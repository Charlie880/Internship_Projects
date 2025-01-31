import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the preprocessed (already standardized) dataset
processed_file_path = "C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 3/processed_air_quality.csv"
df_processed = pd.read_csv(processed_file_path)

# Step 1: Apply PCA to reduce the dataset to two principal components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_processed)

# Step 2: Visualize the transformed data in a 2D plot
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue', edgecolor='k', alpha=0.7)
plt.title("PCA: 2D Visualization of Transformed Data")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig("PCA", dpi=300)

# Step 3: Interpret the results by examining the explained variance ratio
explained_variance = pca.explained_variance_ratio_

# Display the variance explained by each principal component
print(f"Variance Explained by Principal Component 1: {explained_variance[0]:.4f}")
print(f"Variance Explained by Principal Component 2: {explained_variance[1]:.4f}")

# Total variance explained by the two components
total_variance_explained = sum(explained_variance)
print(f"Total Variance Explained by the Two Components: {total_variance_explained:.4f}")