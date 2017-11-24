import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

# dictionary
genders = {'male': 1, 'female': -1}
ranks = {"full": 1, "associate": 0, "assistant": -1}
degrees = {"masters": 1, "doctorate": -1}

width = 100

# data input from ../salary.dat
# convert text input to numeric according to dictionary above


def inp(file):
    with open(file, 'r') as input:
        length = 0
        A = np.array([[0, 0, 0, 0, 0, 0]])  # input layer size of 6
        y = np.array([0])
        for line in input.readlines():
            wordList = re.sub("[^\w]", " ", line).split()
            vect = np.array([0, float(genders[wordList[0]]), ranks[wordList[1]], float(
                wordList[2]), degrees[wordList[3]], float(wordList[4])])  # gender, rank, yr, degree received, yd
            A = np.append(A, [vect], axis=0)
            y = np.append(y, [float(wordList[5])], axis=0)
            length += 1
    return np.delete(A, 0, axis=0), np.delete(y, 0), length

# feed forward into the network
# sigmoid activation network


def feed_forward(A, syn, activ="sigmoid"):
    if activ == "sigmoid":
        l1 = sigmoid(np.dot(A, syn))
        return l1
    elif activ == "linear":
        l1 = np.dot(A, syn0)
        return l1
    else:
        l1 = softplus(np.dot(A, syn))  # Sigmoid activation function nodes
        return l1


# sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# linear function


def linear(x):
    return x

# softplus function


def softplus(x):
    return np.log(1 + np.exp(x))


def elm(A, y, syn0, seed=None, activ="sigmoid", width=1000):
    np.random.seed(seed)

    # func = np.random.rand(width, 1) #randomly assign activation function to the nodes of the network

    # read the training dataset
    # randomized input weights
    h = feed_forward(A, syn0, activ)
    # least square learning on the output weight of random layer
    w = np.linalg.lstsq(h, y)[0]

    # #print('',err)
    return w


def evaluationELM(syn0):
    [A, y, length] = inp('salary.dat')
    w = elm(A, y, syn0=syn0, width=width)

    # read the test dataset
    [A, y, length] = inp("salary_test.dat")
    # feed test data into network
    _y = np.dot(feed_forward(A, syn0, "sigmoid"), w)

    # calculate error
    err = np.abs(np.average(_y - y))
    return(err)


def find_direction(X, number_of_particle):
    best_particle = 0
    best_result = evaluationELM(X[0])
    for i in range(1, number_of_particle):
        curr_result = evaluationELM(X[i])
        if(curr_result < best_result):
            best_result = curr_result
            best_particle = i
    D = X - np.full([number_of_particle, 6, width], X[best_particle])
    # D.fill(np.asscalar(X[best_particle]))
    return [D, best_particle, best_result]


def PSO(number_of_iter, number_of_particle):
    # spawn initial position and velocity
    X = np.random.uniform(-1, 1, size=[number_of_particle, 6, width])
    V = np.random.uniform(-1, 1, size=[number_of_particle, 6, width])
    # iteration:
    Dglobal = np.empty([number_of_particle, 6, width])
    best_result = 1000000
    for iter in range(number_of_iter):
        [D, local_best_particle, local_best_result] = find_direction(
            X, number_of_particle)
        if(local_best_result < best_result):
            Dglobal = D
        V = V + D + Dglobal
        X = X + V
    best_particle = 0
    best_result = evaluationELM(X[0])
    for i in range(1, number_of_particle):
        curr_result = evaluationELM(X[i])
        if(curr_result < best_result):
            best_result = curr_result
            best_particle = i
    return X[best_particle]


# X = np.random.normal(size = [10, 6, 250])
result = np.empty(10)
for i in range(0, 10):
    result[i] = evaluationELM(PSO(200, 6))
print(np.average(result))
print(np.std(result))
