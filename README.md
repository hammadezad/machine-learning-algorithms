# machine-learning-algorithms
# Linear Regression

This Jupyter Notebook demonstrates how to implement and visualize **Linear Regression** using Python, NumPy, and scikit-learn.

---

## Overview

Linear Regression is a fundamental machine learning algorithm for predicting a continuous target variable based on one or more input features. This project shows both a manual implementation using NumPy and a comparison with scikit-learn’s built-in `LinearRegression`, including step-by-step explanations and plots.

---

## Contents

- **Generate synthetic data** for regression
- **Visualize** the data and fitted regression line
- **Manual calculation** of regression weights using the Normal Equation
- **Using scikit-learn** for Linear Regression
- **Plot and compare** predictions from both methods

---

## How to Run

1. **Install required packages** (run this in your terminal):
    ```bash
    pip install numpy matplotlib scikit-learn jupyter
    ```

2. **Start Jupyter Notebook**:
    ```bash
    jupyter notebook Linear_Regression.ipynb
    ```
    - This will open the notebook in your browser.

3. **Run each cell** in the notebook to see the outputs, plots, and model coefficients.

---

## What You’ll See

- Scatter plot of the data points
- Fitted regression line
- Printed coefficients for both manual and sklearn models
- Example usage of both approaches

---

## Code Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Generate example data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Manual implementation (Normal Equation)
def manual_linear_regression(X, y):
    X_b = np.c_[np.ones((len(X), 1)), X]  # add bias
    theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta_best

theta = manual_linear_regression(X, y)
print("Manual regression coefficients:", theta)

# Using scikit-learn
reg = LinearRegression().fit(X, y)
print("Sklearn coefficients:", reg.intercept_, reg.coef_)

# Plot
plt.scatter(X, y, color='blue')
plt.plot(X, reg.predict(X), color='red', linewidth=2, label='Predicted line')
plt.title("Linear Regression Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
