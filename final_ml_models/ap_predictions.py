import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import joblib

# Load the trained model
model_path = r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models\gradient_boosting_model.pkl"
model = joblib.load(model_path)

# Fabricate environmental data for a random year
def fabricate_environmental_data():
    np.random.seed(42)  # For reproducibility
    months = list(range(1, 13))
    base_dust_emissions = np.random.uniform(50, 150, size=12)  # Random baseline
    base_sunlight = np.random.uniform(300, 500, size=12)  # Random baseline
    base_wind_speed = np.random.uniform(5, 15, size=12)  # Random baseline

    practices = {
        "No Treatment": {
            "Dust Emissions": base_dust_emissions,
            "Sunlight": base_sunlight,
            "Wind Speed": base_wind_speed,
        },
        "Organic Material Cover": {
            "Dust Emissions": base_dust_emissions * 0.9 + np.random.uniform(-5, 5, size=12),
            "Sunlight": base_sunlight,
            "Wind Speed": base_wind_speed,
        },
        "Mulch": {
            "Dust Emissions": base_dust_emissions * 0.8 + np.random.uniform(-5, 5, size=12),
            "Sunlight": base_sunlight,
            "Wind Speed": base_wind_speed,
        },
    }

    fabricated_data = []
    for practice, data in practices.items():
        for month in months:
            fabricated_data.append({
                "Year": np.random.randint(2020, 2025),
                "Month": month,
                "Practice": practice,
                "Dust Emissions": data["Dust Emissions"][month - 1],
                "Sunlight": data["Sunlight"][month - 1],
                "Wind Speed": data["Wind Speed"][month - 1],
            })

    return pd.DataFrame(fabricated_data)

# Predict cases for fabricated data
def predict_cases_from_fabricated_data(fabricated_df):
    X = fabricated_df[["Dust Emissions", "Sunlight", "Wind Speed"]]
    fabricated_df["Predicted Cases"] = model.predict(X)
    return fabricated_df

# Plot predictions
def plot_predictions(predictions_df):
    plt.figure(figsize=(12, 6))
    for practice in predictions_df["Practice"].unique():
        practice_data = predictions_df[predictions_df["Practice"] == practice]
        plt.plot(practice_data["Month"], practice_data["Predicted Cases"], label=practice)
    plt.title("Predicted Cases by Agricultural Practice")
    plt.xlabel("Month")
    plt.ylabel("Predicted Cases")
    plt.legend()
    plt.grid(True)
    plot_path = r"C:\Users\skale\PycharmProjects\Science Proj 24-25\final_ml_models\plots\predicted_cases_by_practice.png"
    plt.savefig(plot_path)
    plt.show()

# Main script
fabricated_df = fabricate_environmental_data()
fabricated_df.to_csv("agricultural_practices_data.csv", index=False)
predictions_df = predict_cases_from_fabricated_data(fabricated_df)
predictions_df.to_csv("agricultural_practices_predictions.csv", index=False)
plot_predictions(predictions_df)
