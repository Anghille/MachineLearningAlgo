import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class RegressionModel(object):

    def __init__(self, degree=1, bias=True):
        """Init parameters of the BaseModel Class"""

        # Polynomial
        self.degree = degree

        # Bias 
        self.bias = bias


    def bias(self, X):
        """
        Add bias feature the the matrix
        
        Return
        ----------
        Return an array-like object of size (n_sample, n_feature + 1)
        """
        return np.c_[X, np.ones(X.shape[0])]


    def convert_list(self, X=np.array(None)):
        """
        Return an array-like object

        Parameters
        ----------
        X: array-like, list, tuple or pandas dataframe
        
        bias: boolean, init parameters
            Default to false. If set to true, add a bias feature to the array-like matrix. 

        Return
        ----------
        C: array-like object of size n_sample, n_features
            Used to convert to the right array-like object to assure compatibility
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
        """
        Return y-pred from an array of values X
        """
        return None
        
