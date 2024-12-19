import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import joblib

air_temp_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/AirTemp_HeatIndex_Cleaned.csv")
fine_particulate_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/FineParticulateMatter_Cleaned.csv")
precipitation_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/Precipitation_Cleaned.csv")
sunlight_df = pd.read_csv("../CDC Wonder Data/CDC_Wonder_Filtered_Data/Sunlight_Cleaned.csv")

# Preprocess datasets
datasets = [air_temp_df, fine_particulate_df, precipitation_df, sunlight_df]
for df in datasets:
    if 'County' in df.columns:
        df.drop(columns=['County'], inplace=True)
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month Code'].astype(int)

aligned_datasets = [df[(df['Year'] >= 2003) & (df['Year'] <= 2011)].set_index(['Year', 'Month']) for df in datasets]

cases_df = aligned_datasets[0][['Cases']]

merged_features_df = pd.concat(
    [df.drop(columns=['Cases'], errors='ignore') for df in aligned_datasets],
    axis=1,
    join="inner"
)

final_df = pd.merge(
    merged_features_df.reset_index(),
    cases_df.reset_index(),
    on=['Year', 'Month'],
    how='inner'
).set_index(['Year', 'Month'])

# Encode categorical variables
categorical_columns = final_df.select_dtypes(include=['object']).columns
final_df = pd.get_dummies(final_df, columns=categorical_columns, drop_first=True)

# Fill missing values
final_df.fillna(final_df.mean(), inplace=True)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(final_df.drop(columns=['Cases']))
y = final_df['Cases'].values

# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Gradient Boosting Regressor with Hyperparameter Tuning
gbr = GradientBoostingRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_gbr = grid_search.best_estimator_
print(f"Best Gradient Boosting Parameters: {grid_search.best_params_}")

# Neural Network Regressor for Nonlinear Relationships
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# Evaluate models
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

print("Gradient Boosting Evaluation:")
pred_gbr = evaluate_model(best_gbr, X_test, y_test, "Gradient Boosting Test Set")

print("\nMLP Neural Network Evaluation:")
pred_mlp = evaluate_model(mlp, X_test, y_test, "MLP Test Set")

# Save the better-performing model
best_model = best_gbr if r2_score(y_test, pred_gbr) > r2_score(y_test, pred_mlp) else mlp
joblib.dump(best_model, "valley_fever_best_model.pkl")
print("Best model saved as 'valley_fever_best_model.pkl'")
# Visualize Predicted vs Actual Cases
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Actual Cases", color="blue", marker='o', linestyle='-', alpha=0.7)
plt.plot(pred_mlp, label="Predicted Cases (MLP)", color="red", marker='x', linestyle='--', alpha=0.7)
plt.plot(pred_gbr, label="Predicted Cases (GBR)", color="green", marker='s', linestyle=':', alpha=0.7)
plt.title("Predicted vs Actual Cases (Test Set)")
plt.xlabel("Sample Index")
plt.ylabel("Cases")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("predicted_vs_actual_comparison.png")
plt.show()

# Residual Plot for MLP
residuals_mlp = y_test - pred_mlp
plt.figure(figsize=(10, 6))
plt.scatter(pred_mlp, residuals_mlp, alpha=0.6, color="purple", edgecolors="k")
plt.axhline(y=0, color="red", linestyle="--")
plt.title("Residuals of MLP Predictions")
plt.xlabel("Predicted Cases (MLP)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.grid(True)
plt.tight_layout()
plt.savefig("mlp_residuals.png")
plt.show()

# Save predictions to CSV
comparison_df = pd.DataFrame({
    "Year": X_test[:, 0] if 'Year' in final_df.columns else None,  # Adjust if Year is included in scaled X
    "Month": X_test[:, 1] if 'Month' in final_df.columns else None,  # Adjust if Month is included in scaled X
    "Actual Cases": y_test,
    "Predicted Cases (MLP)": pred_mlp,
    "Predicted Cases (GBR)": pred_gbr
})

# Save the comparison to a CSV file
csv_path = "valley_fever_predictions_comparison.csv"
comparison_df.to_csv(csv_path, index=False)
print(f"Predictions and actual values saved to {csv_path}")
