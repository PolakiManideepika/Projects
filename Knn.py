# K-Nearest Neighbors (K-NN) Regression for Real Estate Price Prediction

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
dataset = pd.read_csv('realestate_priceprediction.csv')

# Clean data: Keep only relevant columns and drop missing values
dataset = dataset[['RM', 'LSTAT', 'MEDV']].dropna()

# Extract features and target
X = dataset[['RM', 'LSTAT']].values  # Now correctly using names
y = dataset['MEDV'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the K-NN Regressor
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Scatter plot of actual vs predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, c='blue', edgecolor='k', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)
plt.title('Actual vs Predicted MEDV')
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.grid(True)
plt.tight_layout()
plt.show()
