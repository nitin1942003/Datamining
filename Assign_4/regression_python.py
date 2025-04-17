import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Optional, List, Union


class LinearRegression:
    def __init__(self):
        self.coefficients = None  # Beta values (including intercept)
        self.r_squared = None     # R-squared value
        self.adj_r_squared = None # Adjusted R-squared
        self.mse = None           # Mean Squared Error
        self.residuals = None     # Residuals from the model
        self.X = None             # Input features
        self.y = None             # Target values
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Store data
        self.X = X
        self.y = y
        
        # Add intercept column (column of 1s)
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        
        # Calculate coefficients using the normal equation: β = (X^T X)^(-1) X^T y
        # This is the analytical solution to the least squares problem
        try:
            X_transpose = X_with_intercept.T
            X_transpose_X = X_transpose.dot(X_with_intercept)
            X_transpose_X_inv = np.linalg.inv(X_transpose_X)
            X_transpose_y = X_transpose.dot(y)
            self.coefficients = X_transpose_X_inv.dot(X_transpose_y)
        except np.linalg.LinAlgError:
            # If matrix is singular, use the pseudoinverse instead
            self.coefficients = np.linalg.pinv(X_with_intercept).dot(y)
        
        # Make predictions on training data
        y_pred = self.predict(X)
        
        # Calculate residuals
        self.residuals = y - y_pred
        
        # Calculate performance metrics
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        # Mean squared error
        self.mse = np.mean(self.residuals ** 2)
        
        # R-squared (coefficient of determination)
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum(self.residuals ** 2)
        self.r_squared = 1 - (ss_residual / ss_total)
        
        # Adjusted R-squared
        if n_samples <= n_features + 1:
            self.adj_r_squared = self.r_squared
        else:
            self.adj_r_squared = 1 - ((1 - self.r_squared) * (n_samples - 1)) / (n_samples - n_features - 1)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' method first.")
        
        # Add intercept column
        X_with_intercept = np.column_stack((np.ones(X.shape[0]), X))
        
        # Make predictions using the learned coefficients
        return X_with_intercept.dot(self.coefficients)
    
    def get_coefficients(self) -> np.ndarray:
        return self.coefficients
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)
    
    def summary(self) -> None:
        if self.coefficients is None:
            print("Model has not been fitted yet.")
            return
        
        print("\n------ Linear Regression Model Summary ------")
        print(f"Number of samples: {self.X.shape[0]}")
        print(f"Number of features: {self.X.shape[1]}")
        
        print("\nCoefficients:")
        print(f"Intercept: {self.coefficients[0]:.4f}")
        for i in range(1, len(self.coefficients)):
            print(f"Feature {i}: {self.coefficients[i]:.4f}")
        
        print("\nPerformance:")
        print(f"Mean Squared Error: {self.mse:.4f}")
        print(f"R-squared: {self.r_squared:.4f}")
        print(f"Adjusted R-squared: {self.adj_r_squared:.4f}")
        print("-----------------------------------------------")
    
    def plot_residuals(self) -> None:
        if self.residuals is None:
            print("Model has not been fitted yet.")
            return
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.predict(self.X), self.residuals)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title('Residuals vs Predicted Values')
        plt.show()
    
    def plot_regression(self, feature_idx: int = 0) -> None:
        if self.coefficients is None:
            print("Model has not been fitted yet.")
            return
        
        if self.X.shape[1] > 1:
            print(f"Plotting regression for feature {feature_idx} (holding other features constant)")
        
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X[:, feature_idx], self.y, alpha=0.7)
        
        # Create line for regression
        x_min, x_max = self.X[:, feature_idx].min(), self.X[:, feature_idx].max()
        x_line = np.linspace(x_min, x_max, 100)
        
        # For multi-features, we use the mean of other features
        if self.X.shape[1] > 1:
            # Create a matrix where each row has the mean value for each feature
            X_mean = np.tile(np.mean(self.X, axis=0), (100, 1))
            # Replace the feature we're plotting with the line values
            X_mean[:, feature_idx] = x_line
            y_line = self.predict(X_mean)
        else:
            # For simple regression, just predict based on the line
            X_line = x_line.reshape(-1, 1)
            y_line = self.predict(X_line)
        
        plt.plot(x_line, y_line, 'r-', linewidth=2)
        plt.xlabel(f'Feature {feature_idx}')
        plt.ylabel('Target')
        plt.title('Linear Regression')
        plt.show()


def load_data_from_csv(filename: str, target_column: int = -1, header: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    try:
        # Read CSV file
        if header:
            data = pd.read_csv(filename)
        else:
            data = pd.read_csv(filename, header=None)
        
        # Convert to numpy array
        data_array = data.values
        
        # Extract features and target
        if target_column == -1:
            X = data_array[:, :-1]
            y = data_array[:, -1]
        else:
            X = np.delete(data_array, target_column, axis=1)
            y = data_array[:, target_column]
        
        return X, y
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return np.array([]), np.array([])


def main():
    # Example 1: Simple linear regression
    print("Example 1: Simple Linear Regression")
    
    # Generate sample data
    np.random.seed(42)
    X_simple = np.random.rand(100, 1) * 10
    y_simple = 2 + 3 * X_simple + np.random.randn(100, 1) * 1.5
    y_simple = y_simple.flatten()  # Convert to 1D array
    
    # Create and fit the model
    model_simple = LinearRegression()
    model_simple.fit(X_simple, y_simple)
    
    # Print model summary
    model_simple.summary()
    
    # Make predictions
    X_new = np.array([[5.5], [7.2]])
    predictions = model_simple.predict(X_new)
    print("\nPredictions for new X values:")
    for i, pred in enumerate(predictions):
        print(f"X = {X_new[i][0]:.2f}, Predicted Y = {pred:.4f}")
    
    # Plot the regression line
    model_simple.plot_regression()
    
    # Example 2: Multiple linear regression
    print("\nExample 2: Multiple Linear Regression")
    
    # Generate sample data
    X_multiple = np.random.rand(100, 3) * 10  # 3 features
    y_multiple = (
        2 +                       # intercept
        3 * X_multiple[:, 0] +    # effect of feature 1
        -1.5 * X_multiple[:, 1] + # effect of feature 2
        2.5 * X_multiple[:, 2] +  # effect of feature 3
        np.random.randn(100) * 2  # noise
    )
    
    # Create and fit the model
    model_multiple = LinearRegression()
    model_multiple.fit(X_multiple, y_multiple)
    
    # Print model summary
    model_multiple.summary()
    
    # Make predictions
    X_new_multiple = np.array([
        [5.5, 2.3, 1.8],
        [7.2, 4.5, 3.2]
    ])
    predictions_multiple = model_multiple.predict(X_new_multiple)
    print("\nPredictions for new X values:")
    for i, pred in enumerate(predictions_multiple):
        print(f"X = {X_new_multiple[i]}, Predicted Y = {pred:.4f}")
    
    # Plot residuals
    model_multiple.plot_residuals()
    
    # Plot regression for first feature
    model_multiple.plot_regression(feature_idx=0)

if __name__ == "__main__":
    main()