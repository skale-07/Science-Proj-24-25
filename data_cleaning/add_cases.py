import pandas as pd

# Load the Land Temp dataset
land_temp = pd.read_csv(r"/CDC Wonder Data/CDC_Wonder_Filtered_Data/LandTemp_Cleaned.csv")

# Load the Cases dataset
cases_data = pd.read_csv(r"/VF Case Data/transformed_cocci_cases.csv")

# Merge the Cases column into the Land Temp dataset based on Year and Month Code
land_temp = land_temp.merge(
    cases_data[['Year', 'Month Code', 'Cases']],
    on=['Year', 'Month Code'],
    how='inner'
)

# Save the updated Land Temp dataset back to the same file
land_temp.to_csv(r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Filtered_Data\LandTemp_Cleaned.csv", index=False)

print("Land Temp dataset updated with Cases successfully.")
file_path = r"/CDC Wonder Data/CDC_Wonder_Filtered_Data/LandTemp_Cleaned.csv"
land_temp = pd.read_csv(file_path)

# Drop redundant columns (keeping only one 'Cases' column)
if 'Cases_x' in land_temp.columns and 'Cases_y' in land_temp.columns:
    land_temp['Cases'] = land_temp['Cases_y']  # Or decide between Cases_x and Cases_y based on your data
    land_temp = land_temp.drop(columns=['Cases_x', 'Cases_y'])

# Save the cleaned dataset
land_temp.to_csv(file_path, index=False)
print("Extra commas and redundant columns removed successfully.")


# land_temp['County'] = land_temp['County'].replace('Maricopa County, AZ', 'Maricopa')

# data = pd.read_csv(land_temp)
# data = data[data['County'] == 'Maricopa County']
# land_temp['Cases'] = 'NaN'
# land_temp.to_csv(r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\CDC_Wonder_Filtered_Data\LandTemp_Cleaned.csv", index=False)
# print(land_temp.head())
# # Ensure both datasets have matching columns for merging
# print(f"Columns in LandTemp: {land_temp.columns}")
# print(f"Columns in Cases dataset: {cases_data.columns}")
#
# # Perform an inner merge on 'Year' and 'Month Code'
# land_temp_with_cases = land_temp.merge(
#     cases_data[['Year', 'Month Code', 'Cases']],  # Selecting only necessary columns
#     on=['Year', 'Month Code'],
#     how='inner'
# )
#
# # Check for missing or misaligned data
# if land_temp_with_cases['Cases'].isnull().any():
#     print("Warning: Missing values found in 'Cases' column after merging.")
# else:
#     print("All 'Cases' values successfully merged.")
#
# # Verify unique counts of `Year` and `Month Code` after the merge
# print("Unique Year and Month Code combinations in LandTemp with Cases:")
# print(land_temp_with_cases[['Year', 'Month Code']].drop_duplicates().shape[0])
#
# # Verify the first few rows of the merged dataset
# print(f"Rows in LandTemp after adding Cases: {land_temp_with_cases.shape[0]}")
# print(land_temp_with_cases.head())
#
# # Save the updated dataset back to a CSV
# output_path = r"C:\Users\skale\PycharmProjects\Science Proj 24-25\CDC Wonder Data\LandTemp_with_Cases.csv"
# land_temp_with_cases.to_csv(output_path, index=False)
# print(f"Land Temp dataset with Cases saved successfully to {output_path}.")
