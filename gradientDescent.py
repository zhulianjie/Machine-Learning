import numpy as np
import random
import matplotlib.pyplot as plot

def GenData(m, slop, variance):
    x = np.zeros(shape=(m,2))
    y = np.zeros(shape=m)
    # the line is y = 2x
    for i in range(0, m):
        x[i][0] = 1
        x[i][1] = i
        y[i] = slop * i
        #+ random.uniform(0,1) * variance
    return x,y

def GradientDescent(x, y, theta, alpha, m, numOfIterations):
    xTrans = x.transpose()
    for i in range(0, numOfIterations):
        hypo = np.dot(x, theta)
        loss = hypo - y
        gradient = np.dot(xTrans, loss) / m
        theta = theta - alpha * gradient
    return theta

x,y = GenData(200, 2, 1)
m, n = np.shape(x)
numIterations = 100000
alpha = 0.0000001
theta = [0, 0]
theta = GradientDescent(x, y, theta, alpha, m, numIterations)
print theta
