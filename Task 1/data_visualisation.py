import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load the dataset
file_path = 'C:/Users/Bibek Paudel/Desktop/Internship_Project/Task 1/cardekho.csv'
car_data = pd.read_csv(file_path)

# Set the style for plots
sns.set(style="whitegrid")

# Create a folder to save plots if it doesn't exist
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

# 1. Distribution of the target variable (car price)
plt.figure(figsize=(8, 5))
sns.histplot(car_data['selling_price'], bins=30, kde=True, color='blue')
plt.title('Distribution of Car Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'car_price_distribution.png'))  # Save plot
plt.close()

# 2. Categorical Feature Analysis
plt.figure(figsize=(12, 5))
sns.countplot(data=car_data, x='seller_type', palette='pastel')
plt.title('Distribution of Seller Types')
plt.xlabel('Seller Type')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'seller_type_distribution.png'))  # Save plot
plt.close()

plt.figure(figsize=(12, 5))
sns.countplot(data=car_data, x='owner', palette='muted', order=car_data['owner'].value_counts().index)
plt.title('Distribution of Owners')
plt.xlabel('Owner')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'owner_distribution.png'))  # Save plot
plt.close()

plt.figure(figsize=(8, 5))
sns.countplot(data=car_data, x='transmission', palette='coolwarm')
plt.title('Distribution of Transmission Types')
plt.xlabel('Transmission')
plt.ylabel('Count')
plt.savefig(os.path.join(output_dir, 'transmission_distribution.png'))  # Save plot
plt.close()

# 3. Relationship Analysis
plt.figure(figsize=(8, 5))
sns.scatterplot(data=car_data, x='mileage(km/ltr/kg)', y='selling_price', hue='transmission')
plt.title('Mileage vs Price')
plt.xlabel('Mileage (km/ltr/kg)')
plt.ylabel('Price')
plt.legend(title='Transmission')
plt.savefig(os.path.join(output_dir, 'mileage_vs_price.png'))  # Save plot
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=car_data, x='engine', y='selling_price', hue='seller_type')
plt.title('Engine Capacity vs Price')
plt.xlabel('Engine (cc)')
plt.ylabel('Price')
plt.legend(title='Seller Type')
plt.savefig(os.path.join(output_dir, 'engine_vs_price.png'))  # Save plot
plt.close()

# 4. Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = car_data.select_dtypes(include=['float64', 'int64']).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))  # Save plot
plt.close()

# Create a folder to save plots
output_dir = "feature_analysis"
os.makedirs(output_dir, exist_ok=True)

# Function to sanitize file names
def sanitize_filename(name):
    return name.replace("(", "_").replace(")", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")

# 1. Numerical Features vs Target (Scatterplots)
numerical_features = ['mileage(km/ltr/kg)', 'engine', 'max_power']
for feature in numerical_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=car_data, x=feature, y='selling_price', hue='transmission')
    plt.title(f'{feature} vs Selling Price')
    plt.xlabel(feature)
    plt.ylabel('Selling Price')
    plt.legend(title='Transmission')
    filename = sanitize_filename(f'{feature}_vs_selling_price.png')  # Sanitize file name
    plt.savefig(os.path.join(output_dir, filename))  # Save figure
    plt.close()

# 2. Categorical Features vs Target (Bar Plots)
categorical_features = ['seller_type', 'owner', 'transmission']
for feature in categorical_features:
    plt.figure(figsize=(8, 5))
    sns.barplot(data=car_data, x=feature, y='selling_price', estimator='mean', errorbar=None, palette='pastel')
    plt.title(f'Average Selling Price by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Average Selling Price')
    filename = sanitize_filename(f'average_selling_price_by_{feature}.png')  # Sanitize file name
    plt.savefig(os.path.join(output_dir, filename))  # Save figure
    plt.close()

# 3. Pairplot for Numerical Relationships
pairplot = sns.pairplot(
    car_data[numerical_features + ['selling_price']],  # Only numerical features
    hue=None,  # Remove hue to avoid categorical conflicts
    palette='coolwarm'
)
pairplot.fig.suptitle('Pairplot of Key Numerical Features', y=1.02)
pairplot_filename = sanitize_filename('pairplot_numerical_features.png')
pairplot.savefig(os.path.join(output_dir, pairplot_filename))  # Save figure
plt.close()