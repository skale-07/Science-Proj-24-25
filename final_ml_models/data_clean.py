import pandas as pd
import numpy as np
import os

# Load the dataset
file_path = "AZMet-Maricopa-Daily-Data-2021-01-01-to-2024-12-01.tsv"
data = pd.read_csv(file_path, sep='\t')

# Convert datetime to a datetime object and extract year and month
data['datetime'] = pd.to_datetime(data['datetime'])
data['Year'] = data['datetime'].dt.year
data['Month'] = data['datetime'].dt.month

# Replace 'NA' and other non-numeric placeholders with NaN
data.replace(to_replace=r"^\s*NA.*$", value=np.nan, regex=True, inplace=True)

# Relevant columns (excluding wind speed)
relevant_columns = [
    'Year', 'Month', 'precip_total_mm', 'sol_rad_total', 'relative_humidity_mean',
    'temp_air_maxF', 'temp_air_minF'
]

# Select relevant columns and ensure numeric types
filtered_data = data[relevant_columns]
numeric_columns = [col for col in filtered_data.columns if col not in ['Year', 'Month']]
filtered_data[numeric_columns] = filtered_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Add Month Code column as a float
filtered_data.insert(2, 'Month Code', filtered_data['Month'].astype(float))
# filtered_data['Month Code'] = filtered_data['Month'].astype(float)

# Compute monthly averages, skipping NaNs
monthly_averages = filtered_data.groupby(['Year', 'Month']).mean().reset_index()

# Rename columns to match the model's expected format
monthly_averages.rename(columns={
    'precip_total_mm': 'Avg Daily Precipitation (mm)',
    'sol_rad_total': 'Avg Daily Sunlight (KJ/m²)',
    'relative_humidity_mean': 'Relative Humidity Mean (%)',
    'temp_air_maxF': 'Avg Daily Max Air Temperature (F)',
    'temp_air_minF': 'Avg Daily Min Air Temperature (F)',
}, inplace=True)

# Save the processed dataset to a CSV
output_dir = r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models"
output_file = os.path.join(output_dir, "Maricopa_Monthly_Averages_2021_2024.csv")

# Create directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save the file
monthly_averages.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
print(monthly_averages.head())
