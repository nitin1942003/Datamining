import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self):
        self.coefficients = None
        self.r_squared = None
        self.adj_r_squared = None 
        self.mse = None
        self.residuals = None
        self.X = None
        self.y = None
        
    def fit(self, X, y):
        # Convert inputs to pandas if they aren't already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        self.X = X
        self.y = y
        
        # Add intercept column
        X_with_intercept = pd.concat([pd.Series(1, index=X.index, name='intercept'), X], axis=1)
        
        try:
            X_transpose = X_with_intercept.T
            X_transpose_X = X_transpose.dot(X_with_intercept)
            X_transpose_X_inv = np.linalg.inv(X_transpose_X)
            X_transpose_y = X_transpose.dot(y)
            self.coefficients = pd.Series(
                X_transpose_X_inv.dot(X_transpose_y),
                index=['intercept'] + list(X.columns)
            )
        except np.linalg.LinAlgError:
            self.coefficients = pd.Series(
                np.linalg.pinv(X_with_intercept).dot(y),
                index=['intercept'] + list(X.columns)
            )
        
        y_pred = self.predict(X)
        self.residuals = y - y_pred
        
        n_samples = len(X)
        n_features = len(X.columns)
        
        self.mse = np.mean(self.residuals ** 2)
        
        y_mean = y.mean()
        ss_total = ((y - y_mean) ** 2).sum()
        ss_residual = (self.residuals ** 2).sum()
        self.r_squared = 1 - (ss_residual / ss_total)
        
        if n_samples <= n_features + 1:
            self.adj_r_squared = self.r_squared
        else:
            self.adj_r_squared = 1 - ((1 - self.r_squared) * (n_samples - 1)) / (n_samples - n_features - 1)
    
    def predict(self, X):
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet. Call 'fit' method first.")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_with_intercept = pd.concat([pd.Series(1, index=X.index, name='intercept'), X], axis=1)
        return X_with_intercept.dot(self.coefficients)
    
    def get_coefficients(self):
        return self.coefficients
    
    def score(self, X, y):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - (u / v)

def main():
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.rand(100, 3) * 10,
        columns=['feature1', 'feature2', 'feature3']
    )
    y = pd.Series(
        2 + 3 * X['feature1'] - 1.5 * X['feature2'] + 2.5 * X['feature3'] + np.random.randn(100) * 2,
        name='target'
    )
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("Coefficients:")
    print(model.get_coefficients())
    print("\nR-squared:", model.r_squared)
    print("Adjusted R-squared:", model.adj_r_squared)
    print("MSE:", model.mse)
    
    X_new = pd.DataFrame({
        'feature1': [5.5, 7.2],
        'feature2': [2.3, 4.5], 
        'feature3': [1.8, 3.2]
    })
    predictions = model.predict(X_new)
    print("\nPredictions for new X values:")
    for i, pred in enumerate(predictions):
        print(f"X = {X_new.iloc[i].values}, Predicted Y = {pred:.4f}")

if __name__ == "__main__":
    main()