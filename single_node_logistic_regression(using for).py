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
    Z=np.zeros((1,ns))
    A=np.zeros((1,ns))
    cost=np.zeros((1,1))
    dw=np.zeros((w.shape))
    db=np.zeros((b.shape))
    for exp in range(0, nd):
        for node in range(0, ns):
            Z[0][exp]+=w[node][0]*X[node][exp]
        Z[0][exp]+=b
        A[0][exp]=sigmoid(Z[0][exp])
        dz=A[0][exp]-Y[0][exp]
        for node in range(0, nd):
            dw[node][0]+=dz*X[node][exp]
        db[0][0]+=dz
        cost[0][0]+=(Y[0][exp]*np.log(A[0][exp])+(1-Y[0][exp])*np.log(1-A[0][exp]))


    for node in range(1,nd):
        dw[node][0]/=ns
    cost[0][0]*=(-1/ns)
    db[0][0]/=ns


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
        for node in range(0, nd):
            w[node][0]-=learning_rate*dw[node][0]
        b[0][0]-=learning_rate*db[0][0]
        if(print_bool==True and i%50==True):
            print("iteration :",i+1," cost=",cost)
    return costs

X,Y=image_array()
X=flatten(X)
X=normalize(X)
ns=X.shape[0]
nd=X.shape[1]
w,b=initialize(X.shape[0])
c1=optimize(w, b, X, Y, ns, nd, 1000, 0.004, True)
x = np.arange(0, 1, 1/5000)
