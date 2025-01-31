import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset correctly, forcing headers to be treated as separate columns
file_path = "C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 3/AirQualityUCI.csv"
df = pd.read_csv(file_path, sep=";", encoding="latin1", skiprows=0)
df = df.iloc[:, 0].str.split(",", expand=True)

# Rename columns based on expected structure
df.columns = ["Date", "Time", "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", 
              "PT08.S2(NMHC)", "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", 
              "PT08.S5(O3)", "T", "RH", "AH"]

print(df.head())

# Rename columns for better readability (if necessary)
df.columns = [col.strip() for col in df.columns]  # Remove any leading/trailing spaces

# Convert Date and Time to a single Datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H.%M.%S', errors='coerce')

# Drop original Date and Time columns
df.drop(['Date', 'Time'], axis=1, inplace=True)

# Replace invalid values (-200) with NaN
df.replace(-200, np.nan, inplace=True)

# Convert all numeric columns to appropriate types
df = df.apply(pd.to_numeric, errors='coerce')

# Fill missing values with column means
df.fillna(df.mean(), inplace=True)

# Select relevant features for clustering (Pollutant Levels & Sensor Readings)
selected_features = [
    'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)', 'PT08.S2(NMHC)', 
    'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 
    'T', 'RH', 'AH'
]
data_selected = df[selected_features]

# Standardize the selected features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_selected)

# Convert standardized data back to DataFrame
df_processed = pd.DataFrame(data_scaled, columns=selected_features)

# Save the cleaned and preprocessed dataset
processed_file_path = "C:/Users/Bibek Paudel/Desktop/Internship_Projects/Task 3/processed_air_quality.csv"
df_processed.to_csv(processed_file_path, index=False)

print("Data preprocessing completed! Processed dataset saved as 'processed_air_quality.csv'.")