
import numpy as np
import pandas as pd
import math as ms
from typing import Union, List, Tuple
from sklearn.preprocessing import PolynomialFeatures

# # : Union[np.ndarray, l, t]
# l = List[Union[float, int]]
# t = Tuple[Union[float, int]]

class LinearRegression:
    """ """
    
    def __init__(self, loss="ols", poly=False, degree=1, verbose=False, X_bias=True):
        self.poly = poly
        self.loss = loss
        self.degree = degree
        self.X_bias = X_bias

        self.weights_ = None

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
        if "ols" in self.loss:
            return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
        else:
            return self.__gradient_descent(X, y, epochs=50)
    

    def __gradient_descent(self, X, y, epochs=50):
        """Compute the gradient descent algorithm and return the weights used to predict y values"""

        def initialize_value(matrix: np.ndarray):
            """Return initialized random values between [-1, 0[ u ]0, 1]
            used for gradient descent (for weights beta and bias)"""

            # Initialize random values between -1/1 for the weights
            initialised_value = np.random.uniform(-1, 1, (matrix.shape[1], 1)) # Generate X.shape[1] weights (should be 3)
            
            # Re-initialize values that are equal to 0 (if any) until no zeros can be found
            # in the initialized weight matrix
            # Why ? Having 0 in the initialize weight would "cancel" this weight at start when 
            # Trying to find the best values. Which would lead to higher error in prediction
            while np.count_nonzero(initialised_value == 0) >= 1:
                for i in range(initialised_value.shape[0]):
                    for j in range(initialised_value.shape[1]):
                        if initialised_value[i,j] == 0:
                            initialised_value[i, j] = np.random.uniform(-1, 1)

            return initialised_value


        def loss(matrix_X, vec_y, weights):
            """Return de quadratic error function used as cost function in linear regression
            Reminder : MSE = 1/2m * SUM ((aX + b) - y)² """

            loss = (1/(2*matrix_X.shape[0])) * (matrix_X @ weights - vec_y).T @ (matrix_X @ weights - vec_y)   # X.T @ X = sum ( X² ) --> X = (matrix_X @ weights - y)

            return loss.tolist()[0][0]


        def gradient(matrix_X, vec_y, weights, alpha=0.01):
            """Compute the gradient descent using weight, X and y as well as the learning rate alpha"""

            derivative = (1/matrix_X.shape[0]) * matrix_X.T @ (matrix_X @ weights - vec_y)
            weights -= alpha * derivative

            return weights

        weights = initialize_value(X)
        if self.verbose:
            for n in range(epochs):
                print("Epoch - ", n)
                print("Loss:", loss(X, y, weights))
                weights = gradient(X, y, weights, alpha=0.0000001)
        else:
            for n in range(epochs):
                weights = gradient(X, y, weights, alpha=0.0000001)
            
        return weights


    def __bias(self, X):
        return np.c_[X, np.ones(X.shape[0])]

    def __convert_list(self, X=np.array(None)):
        """
        self.__convert_list(X)
        ---

        Return a numpy ndarray object

        Parameters
        ---
            - X: a np.ndarray | list | tuple | pandas dataframe
            - bias (False-default, True): add a bias term to the matrix X (bias = matrix(1))
            - dep (False-default, True): 
        """
        try:
            # Convert object to numpy array
            if type(X) != np.ndarray:
                X = np.array(X)
            
            # If simple linear regression, reshape X matrix to a (1,1) vector
            # instead of a flat (1,) vector to avoid errors 
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


    def __poly_transform(self, X):
        """

        """
        # Do the poly transform only if poly set to true and degree > 1
        # if degree == 1, there is no transformation, even if poly == True
        if self.poly and self.degree > 1:
            poly = PolynomialFeatures(self.degree)
            poly.fit(X)
            return X
        else: 
            return X


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

        # Convert X to np.ndarray
        X = self.__convert_list(X)

        # Add polynomial features if specified by ther user with poly == True (and degree > 1)
        X = self.__poly_transform(X)

        # add bias (use it if X isnt normalized)
        if self.X_bias:
            X = self.__bias(X)

        # Convert y to np.ndarray
        y = self.__convert_list(y)

        # Fit linear model weights to data using ols
        self.weights_ = self.__ols(X, y)

        # self.weights_ = self.__gradient_descent(X, y, epochs=50)


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
        # Convert X to np.ndarray
        X = self.__convert_list(X)

        # Add polynomial features if specified by the user using poly = True (and degree > 1)
        X = self.__poly_transform(X)

        # Add bias if specified (best use if data not normalized)
        if self.X_bias:
            X = self.__bias(X)

        return X.dot(self.weights_)
        

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

