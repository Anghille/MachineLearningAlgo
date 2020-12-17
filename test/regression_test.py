import sys
sys.path.append(r"D:\Onedrive\4 - Documents\1 - Formations\Kaggle\MachineLearningAlgo")

import pytest
import numpy as np
import pandas as pd
from linear_reg.regression import LinearRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def x_y():
    """
    Return X_train, X_test, y_train, y_test
    """
    df = pd.read_csv(r"D:\Onedrive\4 - Documents\1 - Formations\Udemy (courses)\The Complete Machine Learning Data Science Course in Python\HousePrice2.csv")
    X = df.iloc[:,:-1].values
    y = df.iloc[:,-1:].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def test_linear_ols_coef(x_y):
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


def test_linear_ols_pred(x_y):
    """
    Check if our predictions are the same as the one found by sklearn
    """

    lin = LinearRegression()
    lin.fit(x_y[0], x_y[2])

    # Predict the values using X_test and out LinearRegression class
    y_pred = lin.predict(x_y[1])
    y_pred = [round(item, 2) for elem in y_pred for item in elem]

    # Access to prediction from sklearn
    y_test = [46371.37368365, 28983.57850126, 34502.91124252]
    y_test = [round(x, 2) for x in y_test]

    assert y_pred == y_test

def test_linear_ols_score(x_y):
    """
    Check if our R² is computed and is the same as skleanr R²
    """ 
    lin = LinearRegression()
    lin.fit(x_y[0], x_y[2])
    y_pred = lin.predict(x_y[1])
    r2 = round(lin.score(x_y[1], x_y[3]), 2)
    
    assert r2 == 0.97

def test_linear_ols_poly_1degree(x_y):
    """
    Check that if we add parameter poly=True but dont specify degree, 
    we keep a multilinear regression with 1-degree features
    """
    lin = LinearRegression(poly=True)
    lin.fit(x_y[0], x_y[2])
    y_pred = lin.predict(x_y[1])
    r2 = round(lin.score(x_y[1], x_y[3]), 2)

    assert r2 == 0.97

def test_linear_ols_poly_2degree(x_y):
    """
    Check that if we add parameter poly=True but dont specify degree, 
    we keep a multilinear regression with 1-degree features
    """
    lin = LinearRegression(poly=True, degree=2)
    lin.fit(x_y[0], x_y[2])
    y_pred = lin.predict(x_y[1])
    r2 = round(lin.score(x_y[1], x_y[3]), 2)
    # print("X_train:", x_y[0].shape, "\ny_train:", x_y[2].shape)
    # print("X_test:", x_y[1].shape, "\ny_test:", x_y[3].shape)

    assert r2 == 0.98

def test_linear_ols_poly_3degree(x_y):
    """
    Check that if we add parameter poly=True but dont specify degree, 
    we keep a multilinear regression with 1-degree features
    """
    lin = LinearRegression(poly=True, degree=3)
    lin.fit(x_y[0], x_y[2])
    y_pred = lin.predict(x_y[1])
    r2 = round(lin.score(x_y[1], x_y[3]), 2)
    # print("X_train:", x_y[0].shape, "\ny_train:", x_y[2].shape)
    # print("X_test:", x_y[1].shape, "\ny_test:", x_y[3].shape)

    assert r2 == 0.80