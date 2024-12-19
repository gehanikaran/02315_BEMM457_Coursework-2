import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Load the cleaned dataset
file_path = 'vgsales_cleaned.csv'  # Replace with the correct path
df = pd.read_csv(file_path)

# Step 1: Prepare Features and Target
# Drop non-numeric or irrelevant columns
df = df.drop(columns=['Name', 'Publisher'], errors='ignore')  # Drop unnecessary columns
X = df.drop(columns=['scaled_Global_Sales'])  # Features
y = df['scaled_Global_Sales']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Initialize Models
models = {
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "LightGBM": LGBMRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# Step 3: Train, Predict, and Evaluate
results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on the test set

    # Calculate Metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Save results
    results[name] = {
        "MAE": mae,
        "MSE": mse,
        "R²": r2,
        "Predicted": y_pred
    }

    # Print Metrics
    print(f"Model: {name}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  R² Score: {r2:.2f}")

# Step 4: Visualize Predictions

# figure 3: Scatter plot 
plt.figure(figsize=(10, 6))
for name, result in results.items():
    plt.scatter(y_test, result["Predicted"], alpha=0.6, label=name)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction Line')
plt.title('Predicted vs Actual Global Sales for Different Models', fontsize=16)
plt.xlabel('Actual Scaled Global Sales', fontsize=14)
plt.ylabel('Predicted Scaled Global Sales', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.show()

# Figure 9: MAE and MSE Comparison of Models
metrics_df = pd.DataFrame({
    "Model": list(results.keys()),
    "MAE": [results[name]["MAE"] for name in results],
    "MSE": [results[name]["MSE"] for name in results],
    "R²": [results[name]["R²"] for name in results]
})

# Figure 9: MAE and MSE Comparison of Models
plt.figure(figsize=(14, 6))
metrics_df.set_index("Model")[["MAE", "MSE"]].plot(kind='bar', edgecolor='black')
plt.title('MAE and MSE Comparison of Models', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Error Value', fontsize=14)
plt.xticks(rotation=45)
plt.grid(alpha=0.5)
plt.show()


