from Image_func import *
from Necessary_Functions import *
import numpy as np
from PIL import Image
import math
import matplotlib.pyplot as plt

np.random.seed(1)


def propagate(w, b, X, Y, ns, nd):
    """
    :param w:
    :param b:
    :param X:
    :param Y:
    :param ns:
    :param nd:
    :return:
    """

    Z=np.dot(w.T,X)+b
    A=sigmoid_numpy(Z)
    A = sigmoid_numpy(np.dot(w.T, X) + b)
    logprobs = np.multiply(np.log(A), Y) + np.multiply((1 - Y), np.log(1 - A))
    cost = - np.sum(logprobs) / nd
    dz=A-Y
    dw=np.dot(X,dz.T)/nd
    db=np.sum(dz)/nd
    db=db.reshape(1,1)

    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost

def optimize(w, b, X, Y, ns, nd, num_iterations, learning_rate, print_bool= False):
    """

    :param w:
    :param b:
    :param X:
    :param Y:
    :param ns:
    :param nd:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    costs=[]
    for i in range(0,num_iterations):
        grads,cost=propagate(w, b, X, Y, ns, nd)
        dw=grads["dw"]
        db=grads["db"]
        costs.append(cost)
        w-=learning_rate*dw
        b-=learning_rate*db
        if(print_bool==True and i%50==True):
            print("iteration :",i," cost=",cost)
    return costs

X,Y=image_array()
ns=X.shape[0]
nd=X.shape[1]
X=flatten(X)
X=normalize(X)
w,b=initialize(X.shape[0])
c1=optimize(w, b, X, Y, ns, nd, 1000 , 0.004, True)