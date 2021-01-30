from itertools import combinations_with_replacement
import numpy as np

__ALL__ = [
    "train_test_split",
    "check_type",
    "initialize_value",
    "polynomial_features"
]

def train_test_split(X, y, train_size=0.8):
    """
    Return 4 array-like to create a train and test set

    Parameters
    ----------
    X : array-like object of size (n_samples, n_features)\n   
    y : array-like object of size (n_samples, 1)\n  
    train_size : a float between 0 and 1. 
        Used to set the size of the train and test sets. 

    Return
    ----------
    x_train: array-like object
    x_test: array-like object
    y-train: array-like object
    y-test: array-like object
    """
    X = check_type(X)
    y = check_type(y)

    X_train = X[:int(train_size*X.shape[0]),:]
    X_test = X[int(train_size*X.shape[0]):,:]

    y_train = y[:int(train_size*y.shape[0])]
    y_test = y[int(train_size*y.shape[0]):]
    return X_train, X_test, y_train, y_test


def check_type(array):
        """
        Return an array-like object

        Parameters
        ----------
        array : array-like, list, tuple or pandas dataframe

        Return
        ----------
        C : array-like object of size n_sample, n_features
            Used to convert to the right array-like object to assure compatibility

        Raise
        ----------
        ValueError: 

        TypeError: 

        IndexError: 
        """
        try:
            # Convert object to numpy array
            if type(array) != np.ndarray:
                array = np.array(array)
            return array

        except ValueError as v:
            print(v)
            raise
        except TypeError as t:
            print(t)
            raise
        except IndexError as i:
            print(i)
            raise


def initialize_value(matrix: np.ndarray):
    """
    Return initialized random values between [-1, 0[ u ]0, 1]
    
    Parameter
    ----------
    Matrix : array-like object of size (n_samples, n_features)

    Return
    ----------
    initialised_value : array like of size (n_features, 1)
    
    """

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



def polynomial_features(array, degree):
    """
    Return the polynomial combination of the features in the provided array

    Parameters
    ----------
    array: an array-like object
    
    degree: the polynomial degree used to compute the combinations. 

    Return
    ---------
    polynomial: array-like object of size (n_samples, n_poly_features)
    """
    array = check_type(array)

    poly = list() # store final polynomial combinations
    combination = ["c"] # store meta_combination
    terms = []

    # Generate numpy features in format ["0","1","2"]
    for feature in range(array.shape[1]):
        combination.append(str(feature))

    # Generate all features-formatted combination
    for comb in combinations_with_replacement(combination, degree): 
        terms.append(comb) 

    terms = list(map(list, terms))

    for i in range(len(terms)): 
        while "c" in terms[i]: 
            terms[i].remove("c") 
    terms=terms[1:]

    # Apply combinations to the numpy arrays
    for comb in terms:
        temp = None
        for element in comb:
            if temp is None:
                temp = array[:,int(element)]
            else:
                temp = temp * array[:,int(element)]
        poly.append(temp)
    
    return np.array(poly).T
        
