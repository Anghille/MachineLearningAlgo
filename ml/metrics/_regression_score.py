import numpy as np

__ALL__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "median_absolute_error",
    "r2_score",
    ]


def r2_score(y_true, y_pred, adjusted=False, n_features=None):
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
    if not adjusted:
        # Compute the normal R2 score
        sstot = np.sum((y_true - np.mean(y_true))**2)
        ssres = np.sum((y_true - y_pred)**2)
        r2_value = 1 - (ssres/sstot)
    else:
        # Compute the adjusted R2 score if adjusted is set to True
        n = y_true.shape[0]
        p = n_features
        sstot = np.sum((y_true - np.mean(y_true))**2)
        ssres = np.sum((y_true - y_pred)**2)
        r_2 = 1 - (ssres/sstot)
        r2_value = 1 - (1 - r_2)* ((n-1)/(n-p-1))

    return r2_value
    

def mean_absolute_error(y_true, y_pred):
    """
    Return the mean absolute error (MAE)

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
    MAE= sum(|y_true - y_pred])/n
    """
    ### ADD raise for using non-array OBJ !!!
    n_samples = y_true.shape[0]
    return np.sum(np.abs(y_true - y_pred))/n_samples


def mean_squared_error(y_true, y_pred):
    """
    Return the mean squared error (MSE)

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
    MSE = (1/n) sum((y_true - y_pred)²)
    """
    ### ADD raise for using non-array OBJ !!!
    n_samples = y_true.shape[0]
    return (1/n_samples)*np.sum((y_true - y_pred)**2)


def median_absolute_error(y_true, y_pred):
    """
    Return the median absolute error (MedAE)

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
    MedAE= median(|y_true - y_pred|])/n
    """
    ### ADD raise for using non-array OBJ !!!
    return np.median(np.abs(y_true - y_pred))