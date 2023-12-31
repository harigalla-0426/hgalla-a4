# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """
    
    # Euclidean distance is calculated as sqrt((x1-y1)^2 + (x2-y2)^2 + ....)
    diff_array = np.array(x1) - np.array(x2)
    euclidean_output = np.sqrt(np.sum(np.power(diff_array, 2)))

    return euclidean_output

    


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    # Manhattan Distance can be calculated as |x1-y1| + |x2-y2| + .... for all n-features
    diff_array = np.array(x1) - np.array(x2)
    manhattan_output = np.sum(np.abs(diff_array))

    return manhattan_output


"""
For the activation functions and their descriptions I referred to: 
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6

And used https://www.wolframalpha.com/ for understanding it's gradient
"""


def identity(x, derivative = False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    identity_output = x

    if derivative:
        identity_output = np.ones(x.shape)

    return identity_output


def sigmoid(x, derivative = False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    sigmoid_output = 1.0/(1.0 + np.exp(-x))

    if derivative:
        sigmoid_output = sigmoid_output - sigmoid_output**2

    return sigmoid_output


def tanh(x, derivative = False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    a = np.exp(x)
    b = np.exp(-x)

    tanh_output = (a-b)/(a+b)

    if derivative:
        tanh_output = 1 - tanh_output**2

    return tanh_output



def relu(x, derivative = False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    relu_output = x * (x > 0)

    if derivative:
        relu_output[relu_output > 0] = 1

    return relu_output


def softmax(x, derivative = False):
    x = np.clip(x, -1e100, 1e100)
    if not derivative:
        c = np.max(x, axis = 1, keepdims = True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis = 1, keepdims = True)))
        # e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        # return e_x / np.sum(e_x, axis=-1, keepdims=True)
    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    # replacing the 0's and 1's if any to a negligible difference, as to avoid div by 0
    # print('here11', y.shape, p.shape)
    np.clip(p, 1e-12, 1-1e-12, out=p)

    comp_p = (1 - p)
    comp_y = (1 - y)

    loss_cal = -(y / p) + comp_y / comp_p

    return loss_cal
    #print('p shape',p.shape)
    # p = np.clip(p, 1e-15, 1 - 1e-15)
    # return - (y / p) + (1 - y) / (1 - p)



def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """
    # encoding_out = []

    # print('output', y)

    # for cl in y:
    #     arr = np.zeros(feature_len)
    #     arr[cl] = 1.0
    #     encoding_out.append(arr)

    # # print('encoding_out', encoding_out)

    # return np.array(encoding_out)

    #index_counter = 0
    #class_map = {}

    #for cl in y:
     #   if cl in class_map:
      #      continue
     #   class_map[cl] = index_counter
    #    index_counter += 1
        
    #encoding_out = []

    #for output_label in y:
    #    arr = np.zeros(index_counter)
    #    arr[class_map[output_label]] = 1.0
    #    encoding_out.append(arr)
    # print('out', y)
    #print('one_hot_encoding', np.array(encoding_out))

    encoding_out = np.zeros((y.size, y.max() + 1))
    encoding_out[np.arange(y.size), y] = 1
    # print('one_hot_encoding', encoding_out)   
    return encoding_out

