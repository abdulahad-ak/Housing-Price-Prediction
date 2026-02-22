"""
===========================================
HOUSING PRICE PREDICTION
California Housing Dataset
AI/ML Internship - Task 3
Author: [Your Name]
===========================================
"""

# ========================================
# IMPORT LIBRARIES
# ========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Create output folders if they don't exist
os.makedirs('outputs/plots', exist_ok=True)

print("=" * 60)
print("   HOUSING PRICE PREDICTION PROJECT")
print("   California Housing Dataset")
print("=" * 60)


# ========================================
# STEP 1: LOAD THE DATASET
# ========================================
print("\n" + "=" * 60)
print("STEP 1: LOADING DATASET")
print("=" * 60)

# Load California Housing Dataset (comes with scikit-learn)
california = fetch_california_housing()

# Create a DataFrame (like Excel table)
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

print("\n✓ Dataset loaded successfully!")
print(f"✓ Total rows: {df.shape[0]}")
print(f"✓ Total columns: {df.shape[1]}")

print("\n--- First 5 Rows of Data ---")
print(df.head())

print("\n--- Column Names and Types ---")
print(df.dtypes)

print("\n--- Statistical Summary ---")
print(df.describe())


# ========================================
# STEP 2: DATA CLEANING
# ========================================
print("\n" + "=" * 60)
print("STEP 2: DATA CLEANING")
print("=" * 60)

# Check for missing values
print("\n--- Checking Missing Values ---")
missing_values = df.isnull().sum()
print(missing_values)
print(f"\nTotal missing values: {missing_values.sum()}")

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
print(f"Duplicate rows: {duplicate_count}")

# Remove duplicates if any
if duplicate_count > 0:
    df = df.drop_duplicates()
    print(f"✓ Removed {duplicate_count} duplicate rows")

# Handle outliers using IQR method
print("\n--- Handling Outliers ---")
columns_to_check = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']

for column in columns_to_check:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    # Count outliers
    outliers = ((df[column] < lower_limit) | (df[column] > upper_limit)).sum()
    print(f"{column}: {outliers} outliers found")
    
    # Cap the outliers (replace extreme values)
    df[column] = df[column].clip(lower=lower_limit, upper=upper_limit)

print("\n✓ Outliers handled successfully!")
print(f"✓ Clean dataset size: {df.shape[0]} rows")


# ========================================
# STEP 3: DATA VISUALIZATION
# ========================================
print("\n" + "=" * 60)
print("STEP 3: DATA VISUALIZATION")
print("=" * 60)

# Plot 1: Distribution of House Prices
print("\nCreating visualizations...")

plt.figure(figsize=(10, 6))
plt.hist(df['MedHouseVal'], bins=50, color='steelblue', edgecolor='black')
plt.xlabel('House Price (in $100,000s)', fontsize=12)
plt.ylabel('Number of Houses', fontsize=12)
plt.title('Distribution of House Prices in California', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/01_price_distribution.png', dpi=150)
plt.close()
print("✓ Saved: 01_price_distribution.png")

# Plot 2: Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation = df.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', linewidths=0.5, square=True)
plt.title('Correlation Between Features', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/plots/02_correlation_heatmap.png', dpi=150)
plt.close()
print("✓ Saved: 02_correlation_heatmap.png")

# Plot 3: Scatter plots of features vs price
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
feature_columns = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                   'Population', 'AveOccup', 'Latitude', 'Longitude']

for i, feature in enumerate(feature_columns):
    row = i // 4
    col = i % 4
    axes[row, col].scatter(df[feature], df['MedHouseVal'], alpha=0.3, s=5, color='blue')
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Price')
    axes[row, col].set_title(f'{feature} vs Price')

plt.tight_layout()
plt.savefig('outputs/plots/03_features_vs_price.png', dpi=150)
plt.close()
print("✓ Saved: 03_features_vs_price.png")

# Plot 4: Geographical Map of Prices
plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                      c=df['MedHouseVal'], cmap='viridis',
                      alpha=0.5, s=5)
plt.colorbar(scatter, label='House Price ($100,000s)')
plt.xlabel('Longitude', fontsize=12)
plt.ylabel('Latitude', fontsize=12)
plt.title('California House Prices by Location', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/plots/04_geographical_map.png', dpi=150)
plt.close()
print("✓ Saved: 04_geographical_map.png")


# ========================================
# STEP 4: FEATURE SELECTION
# ========================================
print("\n" + "=" * 60)
print("STEP 4: FEATURE SELECTION")
print("=" * 60)

# Show correlation with target variable
print("\n--- Correlation with House Price ---")
price_correlation = df.corr()['MedHouseVal'].drop('MedHouseVal')
price_correlation_sorted = price_correlation.sort_values(ascending=False)
print(price_correlation_sorted)

# Select all features for prediction
selected_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                     'Population', 'AveOccup', 'Latitude', 'Longitude']

print(f"\n✓ Selected {len(selected_features)} features for prediction:")
for i, feature in enumerate(selected_features, 1):
    print(f"   {i}. {feature}")

# Separate features (X) and target (y)
X = df[selected_features]
y = df['MedHouseVal']

print(f"\n✓ Features (X) shape: {X.shape}")
print(f"✓ Target (y) shape: {y.shape}")


# ========================================
# STEP 5: SPLIT DATA INTO TRAIN AND TEST
# ========================================
print("\n" + "=" * 60)
print("STEP 5: SPLITTING DATA")
print("=" * 60)

# Split: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n✓ Training set: {X_train.shape[0]} samples (80%)")
print(f"✓ Testing set: {X_test.shape[0]} samples (20%)")


# ========================================
# STEP 6: FEATURE SCALING
# ========================================
print("\n" + "=" * 60)
print("STEP 6: FEATURE SCALING")
print("=" * 60)

# Scale features to have mean=0 and std=1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features scaled using StandardScaler")
print("✓ This helps models perform better")


# ========================================
# STEP 7: TRAIN MACHINE LEARNING MODELS
# ========================================
print("\n" + "=" * 60)
print("STEP 7: TRAINING MODELS")
print("=" * 60)

# MODEL 1: Linear Regression
print("\n--- Training Linear Regression ---")
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_predictions = linear_model.predict(X_test_scaled)
print("✓ Linear Regression model trained!")

# MODEL 2: Random Forest Regressor
print("\n--- Training Random Forest ---")
forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
forest_model.fit(X_train_scaled, y_train)
forest_predictions = forest_model.predict(X_test_scaled)
print("✓ Random Forest model trained!")


# ========================================
# STEP 8: EVALUATE MODELS
# ========================================
print("\n" + "=" * 60)
print("STEP 8: MODEL EVALUATION")
print("=" * 60)

def calculate_metrics(actual, predicted):
    """Calculate evaluation metrics"""
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    return mae, mse, rmse, r2

# Evaluate Linear Regression
lr_mae, lr_mse, lr_rmse, lr_r2 = calculate_metrics(y_test, linear_predictions)

print("\n" + "-" * 40)
print("LINEAR REGRESSION RESULTS:")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {lr_mae:.4f}")
print(f"Mean Squared Error (MSE): {lr_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {lr_rmse:.4f}")
print(f"R² Score: {lr_r2:.4f}")
print(f"Accuracy: {lr_r2 * 100:.2f}%")

# Evaluate Random Forest
rf_mae, rf_mse, rf_rmse, rf_r2 = calculate_metrics(y_test, forest_predictions)

print("\n" + "-" * 40)
print("RANDOM FOREST RESULTS:")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {rf_mae:.4f}")
print(f"Mean Squared Error (MSE): {rf_mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.4f}")
print(f"R² Score: {rf_r2:.4f}")
print(f"Accuracy: {rf_r2 * 100:.2f}%")

# Model Comparison
print("\n" + "=" * 40)
print("MODEL COMPARISON")
print("=" * 40)
print(f"\n{'Metric':<25} {'Linear Reg':<15} {'Random Forest':<15}")
print("-" * 55)
print(f"{'MAE':<25} {lr_mae:<15.4f} {rf_mae:<15.4f}")
print(f"{'RMSE':<25} {lr_rmse:<15.4f} {rf_rmse:<15.4f}")
print(f"{'R² Score':<25} {lr_r2:<15.4f} {rf_r2:<15.4f}")

# Determine best model
if rf_r2 > lr_r2:
    best_model = forest_model
    best_model_name = "Random Forest"
    best_predictions = forest_predictions
    best_r2 = rf_r2
else:
    best_model = linear_model
    best_model_name = "Linear Regression"
    best_predictions = linear_predictions
    best_r2 = lr_r2

print(f"\n{'=' * 40}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"   R² Score: {best_r2:.4f} ({best_r2 * 100:.2f}% accuracy)")
print(f"{'=' * 40}")


# ========================================
# STEP 9: VISUALIZE RESULTS
# ========================================
print("\n" + "=" * 60)
print("STEP 9: VISUALIZING RESULTS")
print("=" * 60)

# Plot 5: Actual vs Predicted Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear Regression Plot
axes[0].scatter(y_test, linear_predictions, alpha=0.5, s=10, color='blue')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Price', fontsize=12)
axes[0].set_ylabel('Predicted Price', fontsize=12)
axes[0].set_title(f'Linear Regression\nR² = {lr_r2:.4f}', fontsize=14)
axes[0].legend()
axes[0].grid(alpha=0.3)

# Random Forest Plot
axes[1].scatter(y_test, forest_predictions, alpha=0.5, s=10, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Actual Price', fontsize=12)
axes[1].set_ylabel('Predicted Price', fontsize=12)
axes[1].set_title(f'Random Forest\nR² = {rf_r2:.4f}', fontsize=14)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/plots/05_actual_vs_predicted.png', dpi=150)
plt.close()
print("✓ Saved: 05_actual_vs_predicted.png")

# Plot 6: Feature Importance
plt.figure(figsize=(10, 6))
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': forest_model.feature_importances_
}).sort_values('Importance', ascending=True)

colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(importance_df)))
plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance (Random Forest)', fontsize=14)
plt.tight_layout()
plt.savefig('outputs/plots/06_feature_importance.png', dpi=150)
plt.close()
print("✓ Saved: 06_feature_importance.png")

# Plot 7: Residual Plot
plt.figure(figsize=(10, 6))
residuals = y_test - best_predictions
plt.scatter(best_predictions, residuals, alpha=0.5, s=10, color='purple')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Price', fontsize=12)
plt.ylabel('Residual (Actual - Predicted)', fontsize=12)
plt.title(f'Residual Plot ({best_model_name})', fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/07_residual_plot.png', dpi=150)
plt.close()
print("✓ Saved: 07_residual_plot.png")

# Plot 8: Model Comparison Bar Chart
plt.figure(figsize=(10, 6))
metrics = ['MAE', 'RMSE', 'R² Score']
linear_values = [lr_mae, lr_rmse, lr_r2]
forest_values = [rf_mae, rf_rmse, rf_r2]

x = np.arange(len(metrics))
width = 0.35

bars1 = plt.bar(x - width/2, linear_values, width, label='Linear Regression', color='steelblue')
bars2 = plt.bar(x + width/2, forest_values, width, label='Random Forest', color='forestgreen')

plt.xlabel('Metric', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.title('Model Comparison', fontsize=14)
plt.xticks(x, metrics)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/08_model_comparison.png', dpi=150)
plt.close()
print("✓ Saved: 08_model_comparison.png")


# ========================================
# STEP 10: SAVE THE MODEL
# ========================================
print("\n" + "=" * 60)
print("STEP 10: SAVING MODEL")
print("=" * 60)

# Save the best model
joblib.dump(best_model, 'outputs/best_model.pkl')
joblib.dump(scaler, 'outputs/scaler.pkl')

print(f"\n✓ Best model saved: outputs/best_model.pkl")
print(f"✓ Scaler saved: outputs/scaler.pkl")


# ========================================
# STEP 11: MAKE A SAMPLE PREDICTION
# ========================================
print("\n" + "=" * 60)
print("STEP 11: SAMPLE PREDICTION")
print("=" * 60)

# Create sample house data
sample_house = {
    'MedInc': 5.5,          # Median income: $55,000
    'HouseAge': 20,          # House age: 20 years
    'AveRooms': 6.0,         # Average rooms: 6
    'AveBedrms': 1.5,        # Average bedrooms: 1.5
    'Population': 1200,      # Population: 1200
    'AveOccup': 3.0,         # Average occupancy: 3
    'Latitude': 34.05,       # Los Angeles area
    'Longitude': -118.25     # Los Angeles area
}

print("\n--- Sample House Features ---")
for feature, value in sample_house.items():
    print(f"   {feature}: {value}")

# Convert to DataFrame
sample_df = pd.DataFrame([sample_house])

# Scale the sample
sample_scaled = scaler.transform(sample_df)

# Make prediction
predicted_price = best_model.predict(sample_scaled)[0]

print(f"\n{'=' * 40}")
print(f"🏠 PREDICTED HOUSE PRICE")
print(f"   ${predicted_price * 100000:,.2f}")
print(f"   (approximately ${predicted_price * 100000 / 1000:.0f}K)")
print(f"{'=' * 40}")


# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)

print(f"""
╔══════════════════════════════════════════════════════════╗
║                    PROJECT SUMMARY                       ║
╠══════════════════════════════════════════════════════════╣
║  Dataset: California Housing Dataset                     ║
║  Total Samples: {len(df):<42}║
║  Features Used: {len(selected_features):<42}║
║  Training Samples: {X_train.shape[0]:<39}║
║  Testing Samples: {X_test.shape[0]:<40}║
║                                                          ║
║  MODELS TRAINED:                                         ║
║    1. Linear Regression (R²: {lr_r2:.4f})                  ║
║    2. Random Forest (R²: {rf_r2:.4f})                      ║
║                                                          ║
║  BEST MODEL: {best_model_name:<35}║
║  ACCURACY: {best_r2 * 100:.2f}%                                       ║
║                                                          ║
║  FILES GENERATED:                                        ║
║    • outputs/best_model.pkl                              ║
║    • outputs/scaler.pkl                                  ║
║    • outputs/plots/01_price_distribution.png             ║
║    • outputs/plots/02_correlation_heatmap.png            ║
║    • outputs/plots/03_features_vs_price.png              ║
║    • outputs/plots/04_geographical_map.png               ║
║    • outputs/plots/05_actual_vs_predicted.png            ║
║    • outputs/plots/06_feature_importance.png             ║
║    • outputs/plots/07_residual_plot.png                  ║
║    • outputs/plots/08_model_comparison.png               ║
╚══════════════════════════════════════════════════════════╝
""")

print("\n✓ All tasks completed!")
print("✓ Check the 'outputs' folder for saved models and plots")
print("\n" + "=" * 60)