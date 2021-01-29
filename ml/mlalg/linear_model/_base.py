import numpy as np

from ...utilities import check_type
from sklearn.preprocessing import PolynomialFeatures

class RegressionModel(object):

    def __init__(self, degree=1, bias=True):
        """Init parameters of the BaseModel Class"""
        
        self.degree = degree
        self.bias = bias


    def bias(self, X):
        """
        Add bias feature the the matrix
        
        Return
        ----------
        Return an array-like object of size (n_sample, n_feature + 1)
        """
        X = check_type(X)
        return np.c_[X, np.ones(X.shape[0])]


    def poly_transform(self, X):
        """
        Add the polynomial features of the matrix

        Return
        ----------
        Return a weight matrix of shape X of shape (n_sample, n_features). 

        Formula
        ----------
        weights_(t) = weights_(t-1) - learning_rate * gradient(MSE w.r.t weights)\n
        gradient(MSE w.r.t weights) = X.T @ [(X @ weights - y)]
        """

        X = check_type(X)
        # Do the poly transform only if poly set to true and degree > 1
        # if degree == 1, there is no transformation, even if poly == True
        if self.degree > 1:
            poly = PolynomialFeatures(self.degree)
            poly.fit(X)
            return X
        else: 
            return X


    def fit(self):
        """Fit the function"""
        return None


    def predict(self):
        """Predict values"""
        return None
        
