import numpy as np
from ml.model.base import BaseModel

# # : Union[np.ndarray, l, t]
# l = List[Union[float, int]]
# t = Tuple[Union[float, int]]



class LinearRegression(BaseModel):
    """
       Linear Regression or Polynomial Regression model

        Parameters
        ----------
        loss: string-like of value "ols" or "gradient"\n
              "ols" is the default value. OLS is used when one want to use Ordinary Least Square method 
              to fit the model. It is recommanded to used the Gradient Descent algorithm when the number of 
              feature exceed 10 due to inverse matrix computation used in OLS.
        
        poly: boolean\n
              Default to False. When set true, polynomial features are added to the feature matrix.
              The degree parameter must be superior to 1 for the polynomials features to be added.
        
        degree: integer\n
                Set the degree of the polynomial regression. For a quadratic polynomial regression, 
                set degree=2. Using a degree of 1, even if poly is set to true, will setup a 
                Linear Regression model. 

        X_bias: boolean\n
                Default to True. Add a bias vector the the feature matrix. Set this parameter to false
                if your data are centered. Otherwise, it is recommanded to let it to default parameters. 
        
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

        Raise
        ----------

        """

    def __init__(self, loss="ols", poly=False, degree=1, X_bias=True, epochs=50, learning_rate=0.01):
        """Init parameters of the LinearRegression Class"""
        # Polynomial
        self.poly = poly
        self.degree = degree

        # Loss function
        self.loss = loss
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        # Bias 
        self.X_bias = X_bias

        # Weights of the regression
        self.weights_ = None
        self.prediction_ = None
    

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

        def __initialize_value(matrix: np.ndarray):
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


        def __gradient(matrix_X, vec_y, weights, learning_rate):
            """Compute the gradient descent using weight, X and y as well as the learning rate alpha"""
            loss = (matrix_X @ weights - vec_y)
            cost = (1/(2*matrix_X.shape[0])) * loss.T @ loss 
            gradient = matrix_X.T @ loss
            weights = weights - learning_rate * (1/matrix_X.shape[0]) * gradient

            return weights, cost.tolist()[0][0]

        self.weights_ = __initialize_value(X)
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

        # Convert X to np.ndarray
        # X = self.__convert_list(X)
        X = super().convert_list(X)

        # Add polynomial features if specified by ther user with poly == True (and degree > 1)
        X = super().poly_transform(X)

        # add bias (use it if X isnt normalized)
        if self.X_bias:
            X = super().bias(X)

        # Convert y to np.ndarray
        y = super().convert_list(y)
  

        # Fit linear model weights to data using ols
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
        # Convert X to np.ndarray
        X = self.convert_list(X)

        # Add polynomial features if specified by the user using poly = True (and degree > 1)
        X = self.poly_transform(X)

        # Add bias if specified (best use if data not normalized)
        if self.X_bias:
            X = self.bias(X)

        self.prediction_ = X.dot(self.weights_)
        

    def score(self, X, y):
        """
        Return the coefficient of determination R²

        Parameters
        ----------
        X: an array-like, list, or tuple of float values\n
        y: an array-like, list, or tuple of float values\n
            X and y must have the same number of observations (X.shape[0] == y.shape[0])

        Exemple
        ----------
        >>> X = [[3, 5, 8], [5, 8, 3], [4, 9, 3], [5, 6, 8]]
        >>> y = [[4], [3], [2], [5]]
        >>> model = LinearRegression()
        >>> model.fit(X, y)
        >>> model.score(X, y)

        Formula
        ----------
            - R2 = 1 - SSres / SStot
            - SSres = SUM (y - y-pred)²
            - SStot = SUM (y - y-mean)²
        """
        try:
            sstot = np.sum((y - np.mean(y))**2)
            ssres = np.sum((y - self.prediction_)**2)

        return 1 - (ssres/sstot)