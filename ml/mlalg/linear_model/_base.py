import numpy as np

from ...utilities import check_type

class RegressionModel(object):

    def __init__(self, degree=1, bias=True):
        """Init parameters of the BaseModel Class"""
        
        self.degree = degree
        self.bias = bias


    def add_bias(self, X):
        """
        Add bias feature the the matrix
        
        Return
        ----------
        Return an array-like object of size (n_sample, n_feature + 1)
        """
        X = check_type(X)
        return np.c_[X, np.ones(X.shape[0])]


    def fit(self):
        """Fit the function"""
        return None


    def predict(self):
        """Predict values"""
        return None
        
