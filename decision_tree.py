# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 12:45:23 2024

@author: manip
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('realestate_priceprediction.csv')
X = dataset.iloc[:, [3,1]].values
y = dataset.iloc[:, 13].values


'''# Load the dataset
file_path = '/mnt/data/dataset_realestate.csv'
dataset = pd.read_csv(file_path)

# Drop the unnamed index column
dataset = dataset.drop(columns=['Unnamed: 0'])

# Define features (X) and target (y)
X = dataset.drop(columns=['MEDV'])
y = dataset['MEDV']'''

# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Regressor model
model = DecisionTreeRegressor(random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)
