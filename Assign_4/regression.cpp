#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

class SimpleLinearRegression {
private:
    // Model parametersa
    double slope;           // m
    double intercept;       // c
    double r_squared;       // R-squared value
    
    // Data
    vector<double> X;  // Input feature
    vector<double> y;  // Target values
    int n_samples;          // Number of observations
    
    // Helper methods
    double calculateMean(const vector<double>& values) {
        double sum = 0.0;
        for (double val : values) {
            sum += val;
        }
        return sum / values.size();
    }
    
    double calculateRSquared(const vector<double>& y_true, const vector<double>& y_pred) {
        double y_mean = calculateMean(y_true);
        double ss_total = 0.0;
        double ss_residual = 0.0;
        
        for (size_t i = 0; i < y_true.size(); ++i) {
            ss_total += pow(y_true[i] - y_mean, 2);
            ss_residual += pow(y_true[i] - y_pred[i], 2);
        }
        
        return 1.0 - (ss_residual / ss_total);
    }
    
public:
    SimpleLinearRegression() : slope(0.0), intercept(0.0), r_squared(0.0), n_samples(0) {}
    
    // Fit the model to the data
    void fit(const vector<double>& X_input, const vector<double>& y_input) {
        if (X_input.empty() || y_input.empty() || X_input.size() != y_input.size()) {
            throw runtime_error("Invalid input data dimensions");
        }
        
        n_samples = X_input.size();
        X = X_input;
        y = y_input;
        
        // Calculate means
        double x_mean = calculateMean(X);
        double y_mean = calculateMean(y);
        
        // Calculate slope (beta1)
        double numerator = 0.0;
        double denominator = 0.0;
        
        for (int i = 0; i < n_samples; ++i) {
            numerator += (X[i] - x_mean) * (y[i] - y_mean);
            denominator += pow(X[i] - x_mean, 2);
        }
        
        slope = numerator / denominator;
        
        // Calculate intercept (beta0)
        intercept = y_mean - slope * x_mean;
        
        // Calculate predictions
        vector<double> y_pred = predict(X);
        
        // Calculate R^2
        r_squared = calculateRSquared(y, y_pred);
    }
    
    // Make predictions for new data
    vector<double> predict(const vector<double>& X_new) {
        if (slope == 0.0 && intercept == 0.0) {
            throw runtime_error("Model not fitted yet");
        }
        
        vector<double> predictions;
        predictions.reserve(X_new.size());
        
        for (double x : X_new) {
            predictions.push_back(intercept + slope * x);
        }
        
        return predictions;
    }
    
    // Get model parameters
    double getSlope() const { return slope; }
    double getIntercept() const { return intercept; }
    double getRSquared() const { return r_squared; }
    
    // Print model summary
    void printSummary() const {
        cout << "\n------ Simple Linear Regression Model Summary ------\n";
        cout << "Number of samples: " << n_samples << endl;
        cout << "\nCoefficients (y = mx + c):\n";
        cout << "Intercept (c): " << intercept << endl;
        cout << "Slope (m): " << slope << endl;
        cout << "\nPerformance:\n";
        cout << "R-squared: " << r_squared << endl;
        cout << "-----------------------------------------------\n";
    }
};

int main() {
    try {
        // Example: Simple Linear Regression with sample data
        cout << "Simple Linear Regression Example\n";
        
        // Sample data (x, y) pairs
        vector<double> X_simple = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};
        vector<double> y_simple = {2.0, 4.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0};
        
        // Create and fit the model
        SimpleLinearRegression model;
        model.fit(X_simple, y_simple);
        
        // Print model summary
        model.printSummary();
        
        // Make predictions for new data
        vector<double> new_X = {11.0, 12.0};
        vector<double> predictions = model.predict(new_X);
        
        cout << "\nPredictions for new X values:\n";
        for (size_t i = 0; i < new_X.size(); ++i) {
            cout << "X = " << new_X[i] << ", Predicted Y = " << predictions[i] << endl;
        }
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}