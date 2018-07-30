import numpy as np
import math
def sigmoid(x):
    """
    :param x: (a vector or a single value)
    :return: sigmoid of all values in x
    note:- math.exp() cant perform stuff on whole matrix/ vector so we need numpy.exp()
    """
    sig=1/(1+math.exp(-x))
    return sig
def sigmoid_numpy(x):
    """
    :param x: (a vector)
    :return: sigmoid of all values in x
    note:- math.exp() cant perform stuff on whole matrix/ vector so we need numpy.exp()
    """
    sig = 1 / (1 + np.exp(-x))
    return sig
def flatten(x):
    """
    :param x: numpy array of shape (no.of examples,60,60,3)
    :process : reshape x to (no.of examples,60*60*3) and then transpose the vector
    :return: a numpy array of shape (60*60*3,no.of examples)
    """
    flattened_image=x.reshape(x.shape[0],-1).T
    return flattened_image
def initialize(dim1):
    """
    :param dim1:size of array
    :return: w vector of size (dim1,1) with random values and b of size(1)
    """
    w=np.random.rand(dim1,1)*0.001
    b=np.zeros((1,1))
    return w,b
def normalize(X):
    """
    :param X: a numpy vector
    :return: a normalized vector
    """
    X=X/255
    return X