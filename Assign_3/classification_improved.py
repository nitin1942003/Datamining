import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeClassifier(criterion='entropy')
        self.feature_names = None
        
    def prepare_data(self, data):
        # Convert categorical variables to numeric using label encoding
        X = data.iloc[:, :-1].copy()
        y = data.iloc[:, -1]
        
        # Store feature names for visualization
        self.feature_names = X.columns.tolist()
        
        # Convert categorical to numeric
        for column in X.columns:
            X.loc[:, column] = pd.Categorical(X[column]).codes
        y = pd.Categorical(y).codes
        
        return X, y
        
    def train(self, train_data):
        X_train, y_train = self.prepare_data(train_data)
        self.model.fit(X_train, y_train)
        return self
    
    def predict(self, test_data):
        X_test, y_test = self.prepare_data(test_data)
        predictions = self.model.predict(X_test)
        
        # Convert numeric predictions back to original labels
        original_classes = pd.Categorical(test_data.iloc[:, -1]).categories
        predictions = [original_classes[p] for p in predictions]
        
        return predictions, test_data.iloc[:, -1].tolist()
    
    def visualize_tree(self):
        plt.figure(figsize=(20,10))
        plot_tree(self.model, 
                 feature_names=self.feature_names,
                 class_names=['no', 'yes'],
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.savefig('decision_tree.png')
        print("\nDecision tree visualization saved as 'decision_tree.png'")

def main():
    # Create training data
    train_data = pd.DataFrame({
        'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 'Overcast', 
                   'Sunny', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Overcast', 'Rainy'],
        'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool', 
                       'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild'],
        'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                    'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
                'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
        'PlayTennis': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes',
                      'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
    })

    # Create test data
    test_data = pd.DataFrame({
        'Outlook': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Overcast', 'Sunny'],
        'Temperature': ['Hot', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Cool'],
        'Humidity': ['High', 'High', 'Normal', 'High', 'Normal', 'Normal', 'High'],
        'Wind': ['Weak', 'Strong', 'Weak', 'Strong', 'Weak', 'Strong', 'Weak'],
        'PlayTennis': ['no', 'no', 'yes', 'no', 'yes', 'yes', 'no']
    })

    print("Training decision tree model...")
    # Create and train model
    model = DecisionTreeModel()
    model.train(train_data)
    
    # Visualize the tree
    model.visualize_tree()
    
    print("\nMaking predictions on test data...")
    # Make predictions on test data
    predictions, actual = model.predict(test_data)
    
    # Print results
    print("\nTest Data:")
    print(test_data.to_string(index=False))
    
    print("\nOriginal Scores:")
    print("  ".join(actual))
    
    print("\nPredicted Scores:")
    print("  ".join(predictions))
    
    # Calculate accuracy
    accuracy = sum(1 for p, a in zip(predictions, actual) if p == a) / len(actual) * 100
    print(f"\nAccuracy on test data: {accuracy}%")

if __name__ == "__main__":
    main() 