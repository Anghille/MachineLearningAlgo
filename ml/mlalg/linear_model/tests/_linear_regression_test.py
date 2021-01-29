import pytest
import numpy as np

from ...utils import train_test_split

from .._linear_regression import LinearRegression



@pytest.fixture
def x_y():
    """
    Return X_train, X_test, y_train, y_test
    """
    X = np.random.random((150,4))
    y = 3*X[:,0] + 2*X[:,1] + 1**X[:,2] + X[:,3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_test, y_train, y_test


def test_linear_ols_regression(x_y):
    """
    Check if our coefficients/weights are the same as the one found by sklearn
    """
    # Call class
    lin = LinearRegression()

    # Fit values from X_train, y_train
    lin.fit(x_y[0], x_y[2])

    # Acces coefficients/weights found by our model
    coefs = lin.weights_
    coefs = coefs.tolist()
    coefs = [round(item, 2) for elem in coefs for item in elem]
    # Compare our coefficients to sklearn coefficients
    assert coefs == [0.53, 5526.54, -58.51, 13163.59]

    y_pred = lin.predict(x_y[1])
    y_pred = [round(item, 2) for elem in y_pred for item in elem]

    # Access to prediction from sklearn
    y_test = [46371.37368365, 28983.57850126, 34502.91124252]
    y_test = [round(x, 2) for x in y_test]

    assert y_pred == y_test


def test_polynomial_ols_regression(x_y):
    """
    Check if our coefficients/weights are the same as the one found by sklearn
    """
    # Call class
    lin = LinearRegression(degree=2)

    # Fit values from X_train, y_train
    lin.fit(x_y[0], x_y[2])

    # Acces coefficients/weights found by our model
    coefs = lin.weights_
    coefs = coefs.tolist()
    coefs = [round(item, 2) for elem in coefs for item in elem]
    # Compare our coefficients to sklearn coefficients
    assert coefs == [0.53, 5526.54, -58.51, 13163.59]

    y_pred = lin.predict(x_y[1])
    y_pred = [round(item, 2) for elem in y_pred for item in elem]

    # Access to prediction from sklearn
    y_test = [46371.37368365, 28983.57850126, 34502.91124252]
    y_test = [round(x, 2) for x in y_test]

    assert y_pred == y_test