import numpy as np

from ...utilities import check_type
from ...utilities import initialize_value
from ...utilities import polynomial_features

from ._base import RegressionModel

__ALL__ = ["LinearRegression"]


class LinearRegression(RegressionModel):
    """
    Linear Regression (Polynomial Regression if degree>1) model

    Parameters
    ----------
    loss: string-like of value "ols" or "gradient"\n
            "ols" is the default value. OLS is used when one want to use Ordinary Least Square method 
            to fit the model. It is recommanded to used the Gradient Descent algorithm when the number of 
            feature exceed 10 due to inverse matrix computation used in OLS.
    
    degree: integer\n
            Determine if the Regression is set to Linear or Polynomial. 
            Using a degree > 1, will transform the regression to a Polynomial Regression model.
            A degree of 1 should be used if one wants to compute a standard Linear Regression.  

    bias: boolean\n
            Default to True. Add a bias vector the the feature matrix. Set this parameter to false
            if your data are centered. Otherwise, it is recommanded to let it to the default parameter. 
    
    epochs: used to set the number of gradient computation when using gradient descent.\n
            Default to 50. Putting a higher number might cause the computation to overload the memory, 
            in case of matrices with many features. Increase with caution. 

    learning_rate: Used during the gradient computation\n
            If the Gradient is returning -inf or +inf, try lower the value using factor 10, 100....
            This parameter control the "speed" of the gradient descent and is considered an hyper-
            parameter. Tweek it to get the optimal value

    Attributes
    ----------
    weights_: None\n
            Weights are computed via the .fit method using the feature matrix and the dependent-vector
            matrix. It is of shape (n_features, 1) if bias is set to False. Otherwise, it is of shape 
            (n_features + 1, 1)                                

    Return
    ----------
    LinearRegression: A LinearRegression object
    """

    def __init__(self, loss="ols", poly=False, degree=1, bias=True, epochs=50, learning_rate=0.01):
        """Init parameters of the LinearRegression Class"""
        # Polynomial
        self.poly = poly
        self.degree = degree

        # Loss function
        self.loss = loss
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Bias 
        self.bias = bias

        # Weights of the regression
        self.weights_ = None
    

    def __ols(self, X, y):
        """
        Compute the weights of the linear model using Ordinary Least Square Method
        
        Return
        ----------
        C: Return a weight matrix of size (n_features, 1)

        Formula
        ----------
        RSS = sum([ax+b - y]²)
        MSE = (1/n) * sum([ax+b - y]²) --> vectorized --> (1/n)(aX - y).T @ (aX - y)
        MSE/a = 0: weights = (X.T @ X)-1 @ (X.T @ y)
        weights = inverse(X.T . X) dot X.T . y
        """

        return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))


    def __gradient_descent(self, X, y, epochs, learning_rate):
        """
        Compute the weights of the linear model using Batch Gradient Descent
        
        Return
        ----------
        C: Return a weight matrix of size (n_features, 1)

        Formula
        ----------
        weights_(t) = weights_(t-1) - learning_rate * gradient(MSE w.r.t weights)
        gradient(MSE w.r.t weights) = X.T @ [(X @ weights - y)]
        """

        def __gradient(matrix_X, vec_y, weights, learning_rate):
            """
            Compute the gradient descent using weight, X and y as well as the learning rate alpha
            """
            loss = (matrix_X @ weights - vec_y)
            cost = (1/(2*matrix_X.shape[0])) * loss.T @ loss 
            gradient = matrix_X.T @ loss
            weights = weights - learning_rate * (1/matrix_X.shape[0]) * gradient

            return weights, cost.tolist()[0][0]

        
        self.weights_ = initialize_value(X)
        for n in range(epochs):
            self.weights_ = __gradient(X, y, self.weights_, learning_rate)[0]
            
        return self.weights_


    def fit(self, X, y):
        """
        Fit the model to the array of features X, y

        Parameters
        ----------
        X : array-like, list, float or int\n
            X must be the same shape as the data the model was trained with. 
        
        Exemple
        ----------
        >>> X_train = [[3, 5, 8], [5, 8, 3], [4, 9, 3], [5, 6, 8]]\n
        >>> print(X_train.shape)
        >>> model.fit(X_train)
        >>> X_test = [[11, 5, 8]]
        >>> model.predict(X_test)
        """

        # Check and convert X,y to np.ndarray
        X = check_type(X)
        y = check_type(y)

        # Add polynomial features if specified by ther user degree > 1
        X = polynomial_features(X, self.degree)

        # add bias (use it if X isnt normalized)
        if self.bias:
            X = super().add_bias(X)
  

        # Fit linear model weights to data using ols or gradient descent
        if "gradient" in self.loss:
            self.weights_ = self.__gradient_descent(X, y, epochs=self.epochs, learning_rate=self.learning_rate)
        else:
            self.weights_ = self.__ols(X, y)


    def predict(self, X):
        """
        Return y-pred from an array of values X

        Parameters
        ----------
        X : array-like, list, float or int\n  
            X must be the same shape as the data the model was trained with. 

        Returns
        ----------
        C : array-like of shape (n_samples, 1)\n   
            Returns the predicted values Ypred computed using features X and
            Ordinary Least Square or Batch Gradient Descent Algorithm

        Exemple
        ----------
        >>> X = [[3, 5, 8], [5, 8, 3], [4, 9, 3], [5, 6, 8]]
        >>> y = [[4], [3], [2], [5]]
        >>> model = LinearRegression()
        >>> model.fit(X, y)
        >>> X_test = [[11, 5, 8]]
        >>> model.predict(X_test)
        """
        # Check and convert X,y to np.ndarray
        print(X)
        X = check_type(X)
        print(X)

        # Add polynomial features if specified by the user using poly = True (and degree > 1)
        X = polynomial_features(X, self.degree)

        # Add bias if specified (best use if data not normalized)
        if self.bias:
            X = super().add_bias(X)

        return X.dot(self.weights_)