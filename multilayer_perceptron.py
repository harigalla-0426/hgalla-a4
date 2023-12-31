# multilayer_perceptron.py: Machine learning implementation of a Multilayer Perceptron classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np

from utils import (cross_entropy, identity, one_hot_encoding, relu, sigmoid,
                   softmax, tanh)


class MultilayerPerceptron:
    """
    A class representing the machine learning implementation of a Multilayer Perceptron classifier from scratch.

    Attributes:
        n_hidden
            An integer representing the number of neurons in the one hidden layer of the neural network.

        hidden_activation
            A string representing the activation function of the hidden layer. The possible options are
            {'identity', 'sigmoid', 'tanh', 'relu'}.

        n_iterations
            An integer representing the number of gradient descent iterations performed by the fit(X, y) method.

        learning_rate
            A float representing the learning rate used when updating neural network weights during gradient descent.

        _output_activation
            An attribute representing the activation function of the output layer. This is set to the softmax function
            defined in utils.py.

        _loss_function
            An attribute representing the loss function used to compute the loss for each iteration. This is set to the
            cross_entropy function defined in utils.py.

        _loss_history
            A Python list of floats representing the history of the loss function for every 20 iterations that the
            algorithm runs for. The first index of the list is the loss function computed at iteration 0, the second
            index is the loss function computed at iteration 20, and so on and so forth. Once all the iterations are
            complete, the _loss_history list should have length n_iterations / 20.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model. This
            is set in the _initialize(X, y) method.

        _y
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.

        _h_weights
            A numpy array of shape (n_features, n_hidden) representing the weights applied between the input layer
            features and the hidden layer neurons.

        _h_bias
            A numpy array of shape (1, n_hidden) representing the weights applied between the input layer bias term
            and the hidden layer neurons.

        _o_weights
            A numpy array of shape (n_hidden, n_outputs) representing the weights applied between the hidden layer
            neurons and the output layer neurons.

        _o_bias
            A numpy array of shape (1, n_outputs) representing the weights applied between the hidden layer bias term
            neuron and the output layer neurons.

    Methods:
        _initialize(X, y)
            Function called at the beginning of fit(X, y) that performs one-hot encoding for the target class values and
            initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.
    """

    def __init__(self, n_hidden = 16, hidden_activation = 'sigmoid', n_iterations = 1000, learning_rate = 0.01):
        # Create a dictionary linking the hidden_activation strings to the functions defined in utils.py
        activation_functions = {'identity': identity, 'sigmoid': sigmoid, 'tanh': tanh, 'relu': relu}

        # Check if the provided arguments are valid
        if not isinstance(n_hidden, int) \
                or hidden_activation not in activation_functions \
                or not isinstance(n_iterations, int) \
                or not isinstance(learning_rate, float):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the MultilayerPerceptron model object
        self.n_hidden = n_hidden
        self.hidden_activation = activation_functions[hidden_activation]
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self._output_activation = softmax
        self._loss_function = cross_entropy
        self._loss_history = []
        self._X = None
        self._y = None
        self._h_weights = None
        self._h_bias = None
        self._o_weights = None
        self._o_bias = None
        self.no_of_features = None
        self.total_output = None

    def _initialize(self, X, y):
        """
        Function called at the beginning of fit(X, y) that performs one hot encoding for the target class values and
        initializes the neural network weights (_h_weights, _h_bias, _o_weights, and _o_bias).

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self.no_of_features = X.shape[1]
        #print('y shape',y.shape)
        self.total_output = y.shape[0]
        
        self._X = X
        self._y = one_hot_encoding(y)
        self.total_output = self._y.shape[1]
        #print('y hot shape', self._y.shape)
        np.random.seed(42)
        
        # Hidden layer
        # Assigning random weights limited by the total output classes
        limit   = 1 / np.sqrt(self.total_output)
        self._h_weights  = np.random.uniform(-limit, limit, (self.no_of_features, self.n_hidden))
        self._h_bias = np.zeros((1, self.n_hidden))
        
        # Output layer
        # Assigning random weights limited by the total number of hidden layers.
        limit   = 1 / np.sqrt(self.n_hidden)
        #print('output size', self.total_output)
        self._o_weights  = np.random.uniform(-limit, limit, (self.n_hidden, self.total_output))
        self._o_bias = np.zeros((1, self.total_output))

    def ret_neuron_compute(self, input_arr, weight_arr, bias, activation_func):
        '''This method the calculates the neuron output when the initial inputs, weights and bias values 
            are passed to it along with the activation function'''
        dot_result = input_arr.dot(weight_arr) + bias
        computed_result = activation_func(dot_result)

        return (dot_result, computed_result)

    def ret_gradient_correction(self, gradient_result, layer_result):
        '''This method is to calculate the gradient loss, which will help in correction of weights and bias values.'''
        grad_w = layer_result.T.dot(gradient_result)
        grad_b = np.sum(gradient_result, axis=0, keepdims=True)

        return (self.learning_rate * grad_w, self.learning_rate * grad_b)

    def convert_output(self, output_hot_encoded):
        '''This method converts the one-hot-encoded output to a one dimensional output.'''
        one_dim_out = []

        for arr in output_hot_encoded:
            one_dim_out.append(np.argmax(arr))

        return one_dim_out

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y and stores the cross-entropy loss every 20
        iterations.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """

        self._initialize(X, y)

        counter = self.n_iterations

        while (counter > 0):

            # hidden layer output and result
            (hidden_params, hidden_layer_result) = self.ret_neuron_compute(self._X, self._h_weights, self._h_bias, self.hidden_activation)

            # hidden layer output and result
            (output_params ,output_layer_result) = self.ret_neuron_compute(hidden_layer_result, self._o_weights, self._o_bias, self._output_activation)
            curr_weights = self._o_weights
            
            # Calculating the gradient loss before backpropagation
            gradient_loss = self._loss_function(self._y, output_layer_result) * self._output_activation(output_params, derivative=True)
            (output_w_err, output_b_err) = self.ret_gradient_correction(gradient_loss, hidden_layer_result)
            self._o_weights  -= output_w_err
            self._o_bias -= output_b_err

            hidden_grd_loss = gradient_loss.dot(curr_weights.T) * self.hidden_activation(hidden_params, derivative= True)
            (hidden_w_err, hidden_b_err) = self.ret_gradient_correction(hidden_grd_loss, self._X)
            # Update weights (by gradient descent)
            # Move against the gradient to minimize loss
            self._h_weights  -= hidden_w_err
            self._h_bias -= hidden_b_err

            counter -= 1

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        hidden_result = self.ret_neuron_compute(X, self._h_weights, self._h_bias, self.hidden_activation)[1]

        final_result = self.ret_neuron_compute(hidden_result, self._o_weights, self._o_bias, self._output_activation)[1]

        output_res = self.convert_output(final_result)
    
        return output_res