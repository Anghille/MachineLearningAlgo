__ALL__ = [
    "train_test_split",
]

def train_test_split(X, y, train_size=0.8):
    X_train = X[:int(train_size*X.shape[0]),:]
    X_test = X[int(train_size*X.shape[0]):,:]

    y_train = y[:int(train_size*y.shape[0])]
    y_test = y[int(train_size*y.shape[0]):]
    return X_train, X_test, y_train, y_test