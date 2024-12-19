import pandas as pd
import os

# Define file paths for each data file and their respective output CSV names
input_files = {
    'CDC Wonder Data/CDCWonderYearlyLandTemp.txt': 'LandTemp_Cleaned.csv',
    'CDC Wonder Data/CDCWonderYearlyAirTemp&HeatIndex.txt': 'AirTemp_HeatIndex_Cleaned.csv',
    'CDC Wonder Data/CDCWonderYearlyFineParticulateMatter.txt': 'FineParticulateMatter_Cleaned.csv',
    'CDC Wonder Data/CDCWonderYearlyPrecipitation.txt': 'Precipitation_Cleaned.csv',
    'CDC Wonder Data/CDCWonderYearlySunlight.txt': 'Sunlight_Cleaned.csv'
}

# Define the output directory
output_directory = 'CDC Wonder Data/CDC_Wonder_Clean_Data'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Function to process each file: ignore 'Notes' column, drop missing values in other columns, and save to CSV
def process_file(input_path, output_filename):
    try:
        df = pd.read_csv(input_path, sep='\t', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, sep='\t', encoding='ISO-8859-1')

    # Drop the 'Notes' column if it exists
    if 'Notes' in df.columns:
        df = df.drop(columns=['Notes'])

    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')  # Convert Year to numeric, setting errors to NaN
        df = df[(df['Year'] >= 2000) & (df['Year'] <= 2015)]  # Filter for years between 2000 and 2015

    # Replace common missing indicators with NaN, then drop rows with NaN in other columns
    df.replace(["Missing", "missing", "", "N/A"], pd.NA, inplace=True)
    df.dropna(how='any', subset=df.columns.difference(['Notes']), inplace=True)

    # Save cleaned data to specified directory if there are remaining rows
    output_path = os.path.join(output_directory, output_filename)
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Processed and saved {output_path} with {len(df)} rows.")
    else:
        print(f"No valid data found in {input_path} after cleaning.")

# Process each file
for input_path, output_filename in input_files.items():
    process_file(input_path, output_filename)