#INITAL CLEANING TO GET ONLY COMMON COUNTY, AND COMMON DATES (NOT COMPLETED - SK 11/6/2024)
# import pandas as pd
# from collections import defaultdict
#
# # Paths to the original CSV files
# file_paths = {
#     "AirTemp_HeatIndex": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\AirTemp_HeatIndex_Cleaned.csv",
#     "FineParticulateMatter": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\FineParticulateMatter_Cleaned.csv",
#     "Precipitation": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\Precipitation_Cleaned.csv",
#     "Sunlight": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\Sunlight_Cleaned.csv"
# }
# import os
# for name, path in file_paths.items():
#     if not os.path.exists(path):
#         print(f"File not found: {path}")
#     else:
#         print(f"File found: {path}")
#
#
# # Load each CSV file into a DataFrame
# dfs = {name: pd.read_csv(path) for name, path in file_paths.items()}
#
# # Extract unique (county, month, year) triples with their respective row indices for each file
# date_county_rows = {name: defaultdict(list) for name in file_paths.keys()}
# for name, df in dfs.items():
#     for idx, row in df.iterrows():
#         county_month_year = (row['County'], row['Month'], row['Year'])
#         date_county_rows[name][county_month_year].append(idx)  # Storing row index (1-based)
#
# # Find common (county, month, year) triples across all files
# common_county_dates = set(date_county_rows['AirTemp_HeatIndex'].keys())
# for name in file_paths.keys():
#     common_county_dates &= set(date_county_rows[name].keys())
# print(common_county_dates)
# # Filter rows in each DataFrame to include only those with the common county, month, and year
# filtered_dfs = {}
# for name, df in dfs.items():
#     # Get rows that match each common (County, Month, Year) triple
#     rows_to_copy = [df.iloc[idx] for date in common_county_dates for idx in date_county_rows[name][date]]
#     filtered_dfs[name] = pd.DataFrame(rows_to_copy)
# # Save the filtered DataFrames as new CSV files with updated criteria
# for name, filtered_df in filtered_dfs.items():
#     file_path = f"{name}_Filtered_by_County_Month_Year.csv"
#     filtered_df.to_csv(file_path, index=False)
#     print(f"Filtered data saved to {file_path}")
#ADDED CASES TO THE ENVIRONMENTAL FILES (COMPLETED - SK 11/12/2024)
import pandas as pd

# Load Valley Fever cases data
vf_cases_path = r"/VF Case Data/cocci_cases.csv"
vf_cases = pd.read_csv(vf_cases_path, index_col="County")

# Reshape vf_cases to long format
vf_cases_long = vf_cases.reset_index().melt(id_vars="County", var_name="Year_Month", value_name="Cases")
vf_cases_long[['Year', 'Month']] = vf_cases_long['Year_Month'].str.split('/', expand=True)
vf_cases_long['Year'] = vf_cases_long['Year'].astype(int)
vf_cases_long['Month Code'] = vf_cases_long['Month'].astype(int)
vf_cases_long.drop(columns=['Year_Month', 'Month'], inplace=True)
# Print sample of reshaped vf_cases_long
print("\nSample vf_cases_long:\n", vf_cases_long.head())

# Define file paths
file_paths = {
    "LandTemp": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDCWonderYearlyLandTemp.txt",
    "AirTemp_HeatIndex": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\AirTemp_HeatIndex_Cleaned.csv",
    "FineParticulateMatter": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\FineParticulateMatter_Cleaned.csv",
    "Precipitation": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\Precipitation_Cleaned.csv",
    "Sunlight": r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Clean_Data\Sunlight_Cleaned.csv"
}
# Iterate over each environmental file and merge with reshaped Valley Fever cases data
for label, file_path in file_paths.items():
    # Load the environmental data file with quote handling
    env_data = pd.read_csv(file_path, delimiter='\t', quotechar='"')

    # Remove quotes from column names if necessary
    env_data.columns = env_data.columns.str.strip().str.replace('"', '')

    # Standardize 'County' column in environmental data
    env_data['County'] = env_data['County'].str.replace(" County, AZ", "").str.strip()

    # Convert 'Year' and 'Month Code' to numeric with error handling
    env_data['Year'] = pd.to_numeric(env_data['Year'], errors='coerce')
    env_data['Month Code'] = pd.to_numeric(env_data['Month Code'], errors='coerce')

    # Drop rows with NaN values in 'Year' or 'Month Code' after conversion
    env_data = env_data.dropna(subset=['Year', 'Month Code'])
    env_data['Year'] = env_data['Year'].astype(int)
    env_data['Month Code'] = env_data['Month Code'].astype(int)

    # Perform the merge
    combined_data = env_data.merge(vf_cases_long, on=['County', 'Year', 'Month Code'], how='left')

    # Rename 'Cases_y' (from vf_cases_long) to 'Cases' and drop 'Cases_x' (if exists)
    if 'Cases_y' in combined_data.columns:
        combined_data.rename(columns={'Cases_y': 'Cases'}, inplace=True)
    if 'Cases_x' in combined_data.columns:
        combined_data.drop(columns=['Cases_x'], inplace=True)

    # Verify the merged columns and sample output
    print("\nColumns in combined_data after merge:", combined_data.columns)
    print(f"Sample of combined data for {label}:\n", combined_data[['County', 'Year', 'Month Code', 'Cases']].head(10))

    # Save the merged file back to its original path
    combined_data.to_csv(file_path, index=False)

print("Valley Fever case data successfully added to all specified files.")
