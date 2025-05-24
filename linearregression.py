#importing the requirements
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 from sklearn.model_selection import train_test_split
 from sklearn.linear_model import LinearRegression
 from sklearn.metrics import mean_squared_error, r2_score
 # Load the dataset
 dataset = pd.read_csv('realestate_priceprediction.csv')
 X =dataset.iloc[:, [2, 10]].values # Assuming these are the
 relevant features
 y =dataset.iloc[:, 13].values # Assuming this is the target variable
 # Split the dataset into training (80%) and testing (20%) sets
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
 random_state=42)
 # Initialize the linear regression model
 model = LinearRegression()
 # Train the model on the training data
 model.fit(X_train, y_train)
 # Makepredictions on the testing data
 y_pred = model.predict(X_test)
 # Evaluate the model
 mse =mean_squared_error(y_test, y_pred)
 23
r2 = r2_score(y_test, y_pred)
 print("Mean Squared Error:", mse)
 print("R-squared:", r2)
 # Plot the actual vs predicted values
 plt.figure(figsize=(10, 6))
 plt.scatter(y_test, y_pred, color='blue')
 plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
 color='red', lw=2) # Diagonal line
 plt.xlabel('Actual Values')
 plt.ylabel('Predicted Values')
 plt.title('Actual vs Predicted Values')
 plt.show()