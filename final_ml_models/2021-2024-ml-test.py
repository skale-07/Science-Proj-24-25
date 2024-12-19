# # Load the model
# import joblib
# import pandas as pd
#
# model = joblib.load("gradient_boosting_model.pkl")
# print("Model loaded successfully")
#
# # Extract the expected feature names from the trained model
# expected_features = model.feature_names_in_
#
# # Load the recent data
# recent_data_path = "Maricopa_Monthly_Averages_2021_2024.csv"
# recent_data = pd.read_csv(recent_data_path)
#
# # Ensure the Month Code column exists
# recent_data['Month Code'] = recent_data['Month'].astype(float)
#
# # Add or rename columns to align with the trained model
# recent_data['Year Code'] = recent_data['Year']
# recent_data['County Code'] = 0  # Default placeholder for County Code
# recent_data['Avg Daily Max Heat Index (F)'] = 0  # Placeholder
# recent_data['Avg Fine Particulate Matter (µg/m³)'] = 0  # Placeholder
#
# # Add season columns
# recent_data['Season_Other'] = 0  # Default placeholder
# recent_data['Season_Summer'] = 0  # Default placeholder
#
# # Ensure column alignment and order with the model's training features
# recent_data_aligned = recent_data[expected_features]
#
# # Fill missing values
# recent_data_aligned.fillna(recent_data_aligned.mean(), inplace=True)
#
# # Predict cases using the model
# predicted_cases = model.predict(recent_data_aligned)
#
# # Add predictions to the recent data
# recent_data['Predicted Cases'] = predicted_cases
#
# # Aggregate the predicted cases by year
# aggregated_cases = recent_data.groupby('Year')['Predicted Cases'].sum().reset_index()
# aggregated_cases.rename(columns={'Predicted Cases': 'Total Predicted Cases'}, inplace=True)
#
# # Save the aggregated predictions to a CSV file
# output_path = "Aggregated_Valley_Fever_Cases_2021_2024.csv"
# aggregated_cases.to_csv(output_path, index=False)
#
# print("Aggregated predictions saved to:", output_path)
# print(aggregated_cases.head())
# Load the model
import joblib
import pandas as pd

model = joblib.load("gradient_boosting_model.pkl")
print("Model loaded successfully")

# Extract the expected feature names from the trained model
expected_features = model.feature_names_in_

# Load the recent data
recent_data_path = "Maricopa_Monthly_Averages_2021_2024.csv"
recent_data = pd.read_csv(recent_data_path)

# Ensure the Month Code column exists
recent_data['Month Code'] = recent_data['Month'].astype(float)

# Add or rename columns to align with the trained model
recent_data['Year Code'] = recent_data['Year']
recent_data['County Code'] = 0  # Default placeholder for County Code
recent_data['Avg Daily Max Heat Index (F)'] = 0  # Placeholder
recent_data['Avg Fine Particulate Matter (µg/m³)'] = 0  # Placeholder

# Add season columns
recent_data['Season_Other'] = 0  # Default placeholder
recent_data['Season_Summer'] = 0  # Default placeholder

# Ensure column alignment and order with the model's training features
recent_data_aligned = recent_data[expected_features]

# Fill missing values
recent_data_aligned.fillna(recent_data_aligned.mean(), inplace=True)

# Predict cases using the model
predicted_cases = model.predict(recent_data_aligned)

# Add predictions to the recent data
recent_data['Predicted Cases'] = predicted_cases

# Save the monthly predictions to a CSV file
monthly_output_path = "Monthly_Valley_Fever_Predictions_2021_2024.csv"
recent_data[['Year', 'Month', 'Predicted Cases']].to_csv(monthly_output_path, index=False)

print("Monthly predictions saved to:", monthly_output_path)

# Aggregate the predicted cases by year
aggregated_cases = recent_data.groupby('Year')['Predicted Cases'].sum().reset_index()
aggregated_cases.rename(columns={'Predicted Cases': 'Total Predicted Cases'}, inplace=True)

# Save the aggregated predictions to a CSV file
aggregated_output_path = "Aggregated_Valley_Fever_Cases_2021_2024.csv"
aggregated_cases.to_csv(aggregated_output_path, index=False)

print("Aggregated predictions saved to:", aggregated_output_path)
print(aggregated_cases.head())

# Debugging: Print monthly predictions for analysis
print("Monthly Predictions:")
print(recent_data[['Year', 'Month', 'Predicted Cases']].head(24))  # Print predictions for 2 years
