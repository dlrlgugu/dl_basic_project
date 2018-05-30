import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()

shape_x = np.shape(X) #(2, 400)
shape_y = np.shape(Y) #(1, 400)
# m = training set = 400
print(shape_x)
print(shape_y)
print(np.shape(X)[0])
"""
- n_x: the size of the input layer
- n_h: the size of the hidden layer (set this to 4) 
- n_y: the size of the output layer
"""
def layer_sizes(X,Y):
    n_x = np.shape(X)[0]
    n_h = 4
    n_y = np.shape(Y)[0]
    return (n_x,n_h,n_y)

X_assess, Y_assess = layer_sizes_test_case()
print(X_assess, Y_assess)
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)

    W1=np.random.randn(n_h,n_x)*0.01
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    assert(W1.shape ==(n_h,n_x))
    assert(b1.shape ==(n_h,1))
    assert(W2.shape ==(n_y,n_h))
    assert(b2.shape ==(n_y,1))

    parameters = {"w1":W1,
                  "b1":b1,
                  "w2":W2,
                  "b2":b2}

    return parameters

n_x,n_h,n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x,n_h,n_y)
print(parameters)


def forward_propagation(x,parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    z1 = np.dot(W1,x)+b1
    A1=np.tanh(z1)

    z2 = np.dot(W2,A1)+b2
    A2=sigmoid(z2)

    assert(A2.shape == (1,x.shape[1]))

    cache = {"z1":z1,
             "A1":A1,
             "z2":z2,
             "A2":A2}

    return A2,cache


X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['z1']) ,np.mean(cache['A1']),
      np.mean(cache['z2']),np.mean(cache['A2']))


def compute_cost(a2,Y,parameters):
    m = Y.shape[1]

    logprobs = np.multiply(np.log(a2),Y)+np.multiply((1-Y),np.log(1-a2))
    cost = -np.sum(logprobs)/m

    cost = np.squeeze(cost)
    assert(isinstance(cost,float))

    return cost

A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]

    W1=parameters["W1"]
    W2=parameters["W2"]

    A1=cache["A1"]
    A2=cache["A2"]

    dz2 = A2-Y
    dW2 = (1/m)*np.dot(dz2,A1.T)
    db2 = np.sum(dz2,axis=1,keepdims=True)*(1/m)

    dz1 = np.multiply(np.dot(W2.T,dz2),(1-np.power(A1,2)))
    dW1 = (1/m)*np.dot(dz1,X.T)
    db1 = (1/m)*np.sum(dz1,axis=1,keepdims=True)

    grads = {"dW1":dW1,
             "db1":db1,
             "dW2":dW2,
             "db2":db2}

    return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))


def update_parameters(parameters,grads,learning_rate=0.1):
    W1=parameters['W1']
    b1=parameters['b1']
    W2=parameters['W2']
    b2=parameters['b2']


    dw1=grads['dW1']
    db1=grads['db1']
    dw2=grads['dW2']
    db2=grads['db2']

    #update parameters
    W1 = W1-learning_rate*dw1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dw2
    b2 = b2-learning_rate*db2

    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}

    return parameters

parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

















    
