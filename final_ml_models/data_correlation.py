import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

# Load datasets
air_temp_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/AirTemp_HeatIndex_Cleaned.csv")
fine_particulate_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/FineParticulateMatter_Cleaned.csv")
precipitation_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/Precipitation_Cleaned.csv")
wind_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/refined_maricopa_wind_data.csv")
sunlight_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/Sunlight_Cleaned.csv")

# Extract 'Cases' column
cases_column = air_temp_df[['Year', 'Month', 'Cases']]

# Filter cases_column for Fine Particulate range (2003–2011)
cases_column = cases_column[(cases_column['Year'] >= 2003) & (cases_column['Year'] <= 2011)]

# Normalize Month values in cases_column to numeric
month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}
cases_column['Month'] = cases_column['Month'].map(month_mapping)

# Normalize other datasets by Year and Month
datasets = [fine_particulate_df, precipitation_df, wind_df, sunlight_df]
for df in datasets:
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month Code'].astype(int)

# Debugging: Check common Year-Month values
common_year_month = set.intersection(
    set(cases_column[['Year', 'Month']].apply(tuple, axis=1)),
    *[set(df[['Year', 'Month']].apply(tuple, axis=1)) for df in datasets]
)
print(f"Common Year-Month values: {len(common_year_month)}")

# Filter datasets to include only common Year-Month values
cases_column = cases_column[cases_column[['Year', 'Month']].apply(tuple, axis=1).isin(common_year_month)]
datasets = [
    df[df[['Year', 'Month']].apply(tuple, axis=1).isin(common_year_month)] for df in datasets
]

# Merge datasets
merged_features = pd.concat([df.set_index(['Year', 'Month']) for df in datasets], axis=1, join='inner')
if 'Cases' in merged_features.columns:
    merged_features.drop(columns=['Cases'], inplace=True)
merged_df = merged_features.join(cases_column.set_index(['Year', 'Month']), how='inner')
print(f"Final Merged DataFrame Shape: {merged_df.shape}")

# Prepare features and target
X = merged_df.drop(columns=['Cases'], errors='ignore')
y = merged_df['Cases'].values.ravel()

# Remove all 'County' and redundant columns upfront
columns_to_drop = [col for col in X.columns if 'County' in col]
print(f"Dropping columns: {columns_to_drop}")
X = X.drop(columns=columns_to_drop, errors='ignore')

# Check and convert all columns in X to numeric
print("Checking and converting all columns in X to numeric...")
X = X.apply(pd.to_numeric, errors='coerce')

# Handle missing values explicitly
if X.isnull().values.any():
    print("Missing values detected in X after conversion. Filling with column means...")
    X.fillna(X.mean(), inplace=True)

# Verify X and y shapes
if X.empty or len(y) == 0:
    raise ValueError("X or y is empty after preprocessing. Ensure the data pipeline is correct.")

# Convert X to NumPy array
print("Converting X to NumPy array...")
X = X.to_numpy()

# Verify that X is numeric
if not np.issubdtype(X.dtype, np.number):
    raise ValueError("X contains non-numeric values. Please check preprocessing.")

# Final shape verification
print(f"Final shapes: X={X.shape}, y={y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=50, max_depth=1),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=50, max_depth=1)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    results[name] = {"MSE": mse, "R2": r2}

# Save results
results_df = pd.DataFrame(results).T
results_df.to_csv("model_comparison_results.csv")
print("Results saved to model_comparison_results.csv")
print(results_df)