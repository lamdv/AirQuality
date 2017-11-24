# pylint: disable=C0103
""" ELM Implementation of Neural Network
"""
import numpy as np
import dataPrepare


inpFile = "./Satellite_data_download/ref.csv"
outFile = "./Satellite_data_download/PRSA_data_2010.1.1-2014.12.31.csv"

A = dataPrepare.inp(inpFile)
y = dataPrepare.out(outFile)

# sigmoid activation network


def feed_forward(A, syn, activ="sigmoid"):
    """feed forward into the network"""
    if activ == "sigmoid":
        l1 = sigmoid(np.dot(A, syn))
    else:
        l1 = softplus(np.dot(A, syn))
    return l1


def sigmoid(x):
    """sigmoid function"""
    return 1 / (1 + np.exp(-x))


def softplus(x):
    """softplus function"""
    return np.log(1 + np.exp(x))


def elm(seed=None, activ="sigmoid", width=1000):
    """
    ELM Training function
    Return a set of input - output weight matrix
    """
    np.random.seed(seed)
    # func = np.random.rand(width, 1) #randomly assign activation function to the nodes of the network

    # read the training dataset
    # randomized input weights
    syn0 = np.random.normal(size=(4, width))
    h = feed_forward(A, syn0, activ)
    # least square learning on the output weight of random layer
    w = np.linalg.lstsq(h, y)[0]

    # calculate error
    err = np.abs(np.average(np.dot(h, w) - y))
    # print('',err)
    return err


[A, y] = dataPrepare.matching(A, y)

# least square learning on the output weight of random layer
# w = np.linalg.lstsq(A, y)[0]

# calculate error
# err = np.abs(np.average(np.dot(A, w) - y))
print(elm())
# # plt.plot(result[:,0], result[:,1], "-", result[:,2], "o")
# plt.plot(result[:,0], result[:,1], '-')
# plt.show()
