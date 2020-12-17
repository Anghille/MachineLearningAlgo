
import numpy as np
import pandas as pd
import math as ms
from typing import Union, List, Tuple

# : Union[np.ndarray, l, t]
l = List[Union[float, int]]
t = Tuple[Union[float, int]]

class LinearRegression:
    """

    """


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
    

    # def __sigmoid(self, X):
    #     """Compute the sigmoid Activation function."""
    #     def sig(x):
    #         return 1 / (1 + ms.exp(-x))
    #     return sig(X)  


    # def __weights_initialization(self, X):
    #     initialised_weights = np.random.rand(X.shape[0], X.shape[1])

    #     # Check that no 0-weights are initialized (otherwize it will not be used in backpropagation)
    #     def remove_zero(initialised_weights):
    #         while np.isin(initialised_weights, 0).any():
    #             for i in range(initialised_weights.shape[0]):
    #                 for j in range(initialised_weights.shape[1]):
    #                     if initialised_weights[i, j] == 0:
    #                         initialised_weights[i, j] = np.random.rand()
    #             return remove_zero(initialised_weights)
    #     remove_zero(initialised_weights)

    #     return initialised_weights

         

    # def __backpropagation(self, X, y):
    #     weights = self.__weights_initialization(X)
    #     bias = np.ones((X.shape[0], 1))

    #     z = weights.dot(X) + bias
    #     activation = self.__sigmoid(z)

    #     return None

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
        # If the polynomial parameter is set to true: 
        # Return a matrix with initial features and
        # product of features and polynomial degrees
        # ex: if X = [x1, x2, x3]
        # then: X -> X = [x1, x2, x3, x1x2, x1x3, x2x3, x1**2, x2**2, x3**3]
        def product(X):  
            """Compute the product of each feature in X
            Return: X matrix with product of each features"""          
            nb_columns = X.shape[1]
            passed_feature = []
            # To make the products x1x2, x1x3, x2x3 ...
            for i in range(nb_columns):
                passed_feature.append(i)
                for f in range(nb_columns):
                    if f in passed_feature:
                        pass
                    else:
                        X = np.c_[X, X[:,f]*X[:,i]]
            return X

        def poly(X, nb_columns):
            """ Compute te polynomial features.\n
            Return: X matrix with polynomial features
            """   
            for feature in X[:, :nb_columns].T:
                for power in range(2, self.degree+1):
                    vec = (feature**power).reshape((feature**power).shape[0], 1)
                    X = np.c_[X, vec]
            return X

        #Initialize global column shape (used in code to avoid X feature duplication)
        col = X.shape[1]

        # Do the poly transform only if poly set to true and degree > 1
        # if degree == 1, there is no transformation, even if poly == True
        if self.poly and self.degree > 1:
            col = X.shape[1]
            X = product(X)
            X = poly(X, col)
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

        # self.weights = self.__backpropagation(X, y)


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


