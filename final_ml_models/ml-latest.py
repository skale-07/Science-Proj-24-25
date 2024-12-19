
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Load datasets
air_temp_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/AirTemp_HeatIndex_Cleaned.csv")
fine_particulate_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/FineParticulateMatter_Cleaned.csv")
precipitation_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/Precipitation_Cleaned.csv")
sunlight_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/Sunlight_Cleaned.csv")

# Drop 'County' column if it exists
datasets = [air_temp_df, fine_particulate_df, precipitation_df, sunlight_df]
for df in datasets:
    if 'County' in df.columns:
        df.drop(columns=['County'], inplace=True)

# Align datasets to 2003–2011
for df in datasets:
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month Code'].astype(int)
    df['Season'] = df['Month'].apply(
        lambda x: 'Summer' if x in [6, 7, 8] else ('Fall' if x in [9, 10, 11] else 'Other')
    )

aligned_datasets = [df[(df['Year'] >= 2003) & (df['Year'] <= 2011)].set_index(['Year', 'Month']) for df in datasets]

# Debugging: Check alignment
print("Aligned Datasets Year-Month Debugging:")
for i, df in enumerate(aligned_datasets):
    print(f"Dataset {i} unique Year-Month pairs: {df.index.unique()}")

# Extract the 'Cases' column separately
if 'Cases' not in aligned_datasets[0].columns:
    raise KeyError("'Cases' column not found in the first dataset.")
cases_df = aligned_datasets[0][['Cases']]

# Debugging: Check cases DataFrame
print("Cases DataFrame Head:\n", cases_df.head())
print("Cases DataFrame Shape:", cases_df.shape)

# Merge datasets
merged_features_df = pd.concat(
    [df.drop(columns=['Cases'], errors='ignore') for df in aligned_datasets],
    axis=1,
    join="inner"
)

# Debugging: Check merged features
print(f"Merged Features Shape (before joining 'Cases'): {merged_features_df.shape}")

# Combine features with the 'Cases' column
final_df = pd.merge(
    merged_features_df.reset_index(),
    cases_df.reset_index(),
    on=['Year', 'Month'],
    how='inner'
).set_index(['Year', 'Month'])

# Remove duplicate column names
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# Debugging: Check final DataFrame
print(f"Final DataFrame Shape: {final_df.shape}")
print("Final DataFrame Columns:", final_df.columns)

if 'Cases' not in final_df.columns:
    print(final_df.head())
    raise KeyError("'Cases' column not found in the final DataFrame after merging.")

# Identify categorical columns for encoding
categorical_columns = final_df.select_dtypes(include=['object']).columns.drop('Cases', errors='ignore')
if not categorical_columns.empty:
    print(f"Encoding categorical columns: {categorical_columns.tolist()}")
    final_df = pd.get_dummies(final_df, columns=categorical_columns, drop_first=True)

# Handle missing values
final_df.fillna(final_df.mean(), inplace=True)

# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = final_df.corr()
correlation_matrix.to_csv(r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models\plots\correlation_matrix.csv")
print("Correlation matrix saved to 'correlation_matrix.csv'")

print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
plt.savefig(r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models\plots\feature_correlation.png")

# Prepare features (X) and target (y)
X = final_df.drop(columns=['Cases'], errors='ignore')  # Features
y = final_df['Cases'].values.ravel()  # Target variable

# Ensure consistent lengths for X and y
if X.shape[0] != len(y):
    print(f"Debugging X and y shapes: X={X.shape}, y={len(y)}")
    raise ValueError("Inconsistent samples between X and y.")

# Train-test-validation split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Drop Year Code and Month Code to test their impact
# X = final_df.drop(columns=['Cases', 'Year Code', 'Month Code'], errors='ignore')  # Features
# y = final_df['Cases'].values.ravel()  # Target variable
#
# # Train-test-validation split
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train Gradient Boosting Regressor
model = GradientBoostingRegressor(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=1,
    random_state=42,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features=0.8
)
model.fit(X_train, y_train)

# Feature Importance Bar Graph
importances = model.feature_importances_
sorted_indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices], align="center")
plt.xticks(range(len(importances)), X.columns[sorted_indices], rotation=90)
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
plt.savefig(r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models\plots\feature_importances.png")

# Evaluate the model
def evaluate_model(model, X, y, dataset_name):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"{dataset_name} Results:")
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2 Score: {r2}")
    return predictions

print("Training Set Evaluation:")
pred_train = evaluate_model(model, X_train, y_train, "Training")

print("\nValidation Set Evaluation:")
pred_val = evaluate_model(model, X_val, y_val, "Validation")

print("\nTest Set Evaluation:")
pred_test = evaluate_model(model, X_test, y_test, "Test")

# Line Graph: Predicted vs. Actual
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Cases", color="blue")
plt.plot(pred_test, label="Predicted Cases", color="red", linestyle="--")
plt.title("Predicted vs. Actual Cases (Test Set)")
plt.xlabel("Sample Index")
plt.ylabel("Cases")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models\plots\predicted_vs_actual_test_set.png")
# Save predictions
comparison_df = pd.DataFrame({
    "Year": X_test.index.get_level_values("Year"),
    "Month": X_test.index.get_level_values("Month"),
    "Actual Cases": y_test,
    "Predicted Cases": pred_test
})
comparison_df.to_csv("test_set_comparison_with_dates.csv", index=False)
print("Test set comparison saved to test_set_comparison_with_dates.csv")

# Save the model
joblib.dump(model, r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models\gradient_boosting_model.pkl")
print("Model saved to 'gradient_boosting_model.pkl'")
