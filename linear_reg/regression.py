
import numpy as np
import pandas as pd
import math as ms
from typing import Union, List, Tuple


l = List[Union[float, int]]
t = Tuple[Union[float, int]]

class LinearRegression:
    """

    """

    def __init__(self, loss="ols", poly=False, degree=1, verbose=False):
        self.poly = poly
        self.loss = loss
        self.degree = degree
        self.weights = None
        self.verbose = verbose
        if self.verbose:
            self.__model_display() 

    def __model_display(self):
        print(f"LinearRegression(Loss='{self.loss}', poly={self.poly}, degree={self.degree}, verbose={self.verbose})")

    def __ols(self, X, y):
        """
        Internal class method
        ---
        Compute the weights of the linear model using Ordinary Least Square Method
        
        ---
        Return
        ---
        Return a weight matrix of shape X

        ---
        Formula
        ---
        weights = inverse(X.T . X) dot X.T . y
        """
        return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
    
    def __sigmoid(self, X):
        """Compute the sigmoid Activation function."""
        def sig(x):
            return 1 / (1 + ms.exp(-x))
        return sig(X)

    def __weights_initialization(self, X):
        initialised_weights = np.random.rand(X.shape[0], X.shape[1])

        # Check that no 0-weights are initialized (otherwize it will not be used in backpropagation)
        def remove_zero(initialised_weights):
            while np.isin(initialised_weights, 0).any():
                for i in range(initialised_weights.shape[0]):
                    for j in range(initialised_weights.shape[1]):
                        if initialised_weights[i, j] == 0:
                            initialised_weights[i, j] = np.random.rand()
                return remove_zero(initialised_weights)
        remove_zero(initialised_weights)

        return initialised_weights

    def __backpropagation(self, X, y):
        weights = self.__weights_initialization(X)
        bias = np.ones((X.shape[0], 1))

        z = weights.dot(X) + bias
        activation = self.__sigmoid(z)

        


        return


    # def __shape(self, X, y):
    #     return
    
    
    def __convert_list(self, X, bias=False):
        try:
            X = np.array(X)
            if bias:
                X = np.c_[X, np.ones(X.shape[0])]
            if len(X.shape) < 2:
                X = X.reshape(X.shape[0], 1)
            return X

        except ValueError as v:
            print(v)
            raise
        except TypeError as t:
            print(t)
            raise
        except IndexError as i:
            print(i)
            raise

    # : Union[np.ndarray, l, t]
    def fit(self, X, y):
        """
        ---
        model.fit(X, y)
        ---
            - Fit the model to the array of features X, y
        ---
        Arguments: (X)\n
            - X (type: np.array, list, float, int): a numpy array, list, float or int
                 X must be the same shape as the data the model was trained with
        ---
        Exemple:\n
        >>> X_train = [[3, 5, 8], [5, 8, 3], [4, 9, 3], [5, 6, 8]]\n
        >>> print(X_train.shape)
        >>> model.fit(X_train)
        >>> X_test = [[11, 5, 8]]
        >>> model.predict(X_test)
        """
        X = self.__convert_list(X, bias=True)
        y = self.__convert_list(y, bias=False)

        if "ols" in self.loss:
            self.weights = self.__ols(X, y)
        else: 
            self.weights = self.__backpropagation(X, y)


    def predict(self, X):
        """
        ---
        model.predict(X)
        ---
            - Return y-pred from an array of values X
        ---
        Arguments: (X)\n
            - X (type: np.array, list, float, int): a numpy array, list, float or int\n
            X must be the same shape as the data the model was trained with
        ---
        Exemple:\n
        >>> X = [[3, 5, 8], [5, 8, 3], [4, 9, 3], [5, 6, 8]]
        >>> y = [[4], [3], [2], [5]]
        >>> model = LinearRegression()
        >>> model.fit(X, y)
        >>> X_test = [[11, 5, 8]]
        >>> model.predict(X_test)
        """
        X = self.__convert_list(X, bias=True)
        return X.dot(self.weights)
        

    def score(self, X, y):
        """
        ---
        model.score(X, y)
        ---s
            - Return the coefficient of determination R²
        ---
        Arguments: (X, y)\n
            - X (type: np.ndarray, list, tuple): a np.ndarray, list or tuple of float values
            - y (type: np.ndarray, list, tuple): a np.ndarray, list or tuple of float values\n
            X and y must have the same number of observations (X.shape[0] == y.shape[0])
        ---
        Exemple:\n
        >>> X = [[3, 5, 8], [5, 8, 3], [4, 9, 3], [5, 6, 8]]
        >>> y = [[4], [3], [2], [5]]
        >>> model = LinearRegression()
        >>> model.fit(X, y)
        >>> model.score(X, y)

        ---
        Formula:\n
            - R2 = 1 - SSres / SStot
            - SSres = SUM (y - y-pred)²
            - SStot = SUM (y - y-mean)²
        """
        sstot = np.sum((y - np.mean(y))**2)
        ssres = np.sum((y - self.predict(X))**2)

        return 1 - (ssres/sstot)

    def __standard_deviation(self, X):
        """
        """
        return X.std(axis=0)

    def __covariance(self, X, y):
        """
        """
        return np.cov(X, y.T)

    def __correlation(self, X, y):
        """
        """
        return np.corrcoef(X.T, y.T)

    def check_assumptions(self, X, y):
        """
        Return the assumption of multilinear regression
        """
        if self.__correlation(X,y).any() < np.abs(0.25):
            print("Some features are not correlated to y")
        else: 
            return 0

