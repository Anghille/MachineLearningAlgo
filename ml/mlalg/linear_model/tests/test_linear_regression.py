import pytest
import numpy as np

from sklearn.linear_model import LinearRegression as LReg
from sklearn.preprocessing import PolynomialFeatures

from ....utilities import train_test_split

from .._linear_regression import LinearRegression


def generate_data():
    """
    Return X_train, X_test, y_train, y_test
    """
    np.random.seed(0)
    X = np.random.random((150,4))
    y = 3*X[:,0] + 2*X[:,1] + 1**X[:,2] + X[:,3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    return X_train, X_test, y_train, y_test


def sklearn_linear(degree):
    X_train, X_test, y_train, y_test = generate_data()

    if degree > 1:
        pol = PolynomialFeatures(degree, include_bias=False)
        X_train = pol.fit_transform(X_train)
        X_test = pol.fit_transform(X_test)

    reg = LReg()
    reg.fit(X_train, y_train)


    pred = reg.predict(X_test).tolist()
    pred = [round(p, 2) for p in pred]

    weights = reg.coef_.tolist()
    weights.append(1.0) # Add bias
    weights = [round(p, 2) for p in weights]

    return pred, weights

def test_linear_ols_regression():
    """
    Check if our coefficients/weights are the same as the one found by sklearn
    Using non-poly and OLS
    """
    degree=1
    X_train, X_test, y_train, y_test = generate_data()
    sk_pred, sk_weights = sklearn_linear(degree=degree)


    # Call class
    lin = LinearRegression(bias=True, degree=1)

    # Fit values from X_train, y_train
    lin.fit(X_train, y_train)

    # Acces coefficients/weights found by our model
    coefs = lin.weights_
    coefs = coefs.tolist()
    coefs = [round(elem, 2) for elem in coefs]
    # Compare our coefficients to sklearn coefficients
    assert coefs == sk_weights

    y_pred = lin.predict(X_test)
    y_pred = [round(item, 2) for item in y_pred]
    assert y_pred == sk_pred


def test_polynomial_ols_regression():
    """
    Check if our coefficients/weights are the same as the one found by sklearn
    Using polynomial and OLS
    """
    degree = 4
    X_train, X_test, y_train, y_test = generate_data()
    sk_pred, sk_weights = sklearn_linear(degree=degree)

    # Call class
    lin = LinearRegression(bias=True, degree=degree)

    # Fit values from X_train, y_train
    lin.fit(X_train, y_train)

    # Acces coefficients/weights found by our model vs sklearn
    coefs = lin.weights_
    coefs = coefs.tolist()
    coefs = [round(item, 2) for item in coefs]
    assert coefs == sk_weights

    y_pred = lin.predict(X_test)
    y_pred = [round(item, 2) for item in y_pred]
    assert y_pred == sk_pred



def test_linear_gradient_regression():
    """
    Check if our coefficients/weights are the same as the one found by sklearn
    Using Non-poly and Batch Gradient Descent
    """
    degree=1
    X_train, X_test, y_train, y_test = generate_data()
    sk_pred, sk_weights = sklearn_linear(degree=degree)


    # Call class
    lin = LinearRegression(bias=True, degree=1, loss="gradient")

    # Fit values from X_train, y_train
    lin.fit(X_train, y_train)

    # Acces coefficients/weights found by our model
    coefs = lin.weights_
    coefs = coefs.tolist()
    coefs = [round(elem, 2) for elem in coefs]
    # Compare our coefficients to sklearn coefficients
    assert coefs == sk_weights

    y_pred = lin.predict(X_test)
    y_pred = [round(item, 2) for item in y_pred]
    assert y_pred == sk_pred


def test_polynomial_gradient_regression():
    """
    Check if our coefficients/weights are the same as the one found by sklearn
    Using polynomial and Batch Gradient Descent
    """
    degree = 4
    X_train, X_test, y_train, y_test = generate_data()
    sk_pred, sk_weights = sklearn_linear(degree=degree)

    # Call class
    lin = LinearRegression(bias=True, degree=degree, loss="gradient")

    # Fit values from X_train, y_train
    lin.fit(X_train, y_train)

    # Acces coefficients/weights found by our model vs sklearn
    coefs = lin.weights_
    coefs = coefs.tolist()
    coefs = [round(item, 2) for item in coefs]
    assert coefs == sk_weights

    y_pred = lin.predict(X_test)
    y_pred = [round(item, 2) for item in y_pred]
    assert y_pred == sk_pred