# k_nearest_neighbors.py: Machine learning implementation of a K-Nearest Neighbors classifier from scratch.
#
# Submitted by: [enter your full name here] -- [enter your IU username here]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff

import numpy as np
from utils import euclidean_distance, manhattan_distance


class KNearestNeighbors:
    """
    A class representing the machine learning implementation of a K-Nearest Neighbors classifier from scratch.

    Attributes:
        n_neighbors
            An integer representing the number of neighbors a sample is compared with when predicting target class
            values.

        weights
            A string representing the weight function used when predicting target class values. The possible options are
            {'uniform', 'distance'}.

        _X
            A numpy array of shape (n_samples, n_features) representing the input data used when fitting the model and
            predicting target class values.

        _y
            A numpy array of shape (n_samples,) representing the true class values for each sample in the input data
            used when fitting the model and predicting target class values.

        _distance
            An attribute representing which distance metric is used to calculate distances between samples. This is set
            when creating the object to either the euclidean_distance or manhattan_distance functions defined in
            utils.py based on what argument is passed into the metric parameter of the class.

    Methods:
        fit(X, y)
            Fits the model to the provided data matrix X and targets y.

        predict(X)
            Predicts class target values for the given test data matrix X using the fitted classifier model.

    Reference video to understand the algorithm: https://www.youtube.com/watch?v=4HKqjENq9OU
    """

    def __init__(self, n_neighbors = 5, weights = 'uniform', metric = 'l2'):
        # Check if the provided arguments are valid
        if weights not in ['uniform', 'distance'] or metric not in ['l1', 'l2'] or not isinstance(n_neighbors, int):
            raise ValueError('The provided class parameter arguments are not recognized.')

        # Define and setup the attributes for the KNearestNeighbors model object
        self.n_neighbors = n_neighbors
        self.weights = weights
        self._X = None
        self._y = None
        self._distance = euclidean_distance if metric == 'l2' else manhattan_distance

    def fit(self, X, y):
        """
        Fits the model to the provided data matrix X and targets y.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the input data.
            y: A numpy array of shape (n_samples,) representing the true class values for each sample in the input data.

        Returns:
            None.
        """
        # print(' ')
        # print('X, y value shape', X.shape, y.shape)
        # print(' ')
        # print('neighbors and weights', self.n_neighbors, self.weights)

        # save the training input data and corresponding classes
        self._X = X
        self._y = y

    def predict(self, X):
        """
        Predicts class target values for the given test data matrix X using the fitted classifier model.

        Args:
            X: A numpy array of shape (n_samples, n_features) representing the test data.

        Returns:
            A numpy array of shape (n_samples,) representing the predicted target class values for the given test data.
        """
        # print(' ')
        # print('predict shape X', X.shape)

        predicted_classes_arr = []

        for x_test_sample in X:
            predicted_classes_arr.append(self.return_predicted_class(x_test_sample))

        return predicted_classes_arr

    def return_predicted_class(self, x_train):
        '''
        This method will predict the class for a given specific test sample by finding the distances
        between sample to all the other samples in training data to by finding the best match w.r.t the
        proximity and the weights too as required.

        Args:
            x_train: A numpy array of shape (n_features) representing a single test sample.
        '''

        cal_dist_arr = []
        for x_test in self._X:
            cal_dist_arr.append(self._distance(x_train, x_test))

        # print('cal_dist_arr', cal_dist_arr)

        sorted_dist_arr = np.argsort(cal_dist_arr)
        k_nearest_indices = sorted_dist_arr[:self.n_neighbors]

        k_nearest_class_labels = []
        for index in k_nearest_indices:
            k_nearest_class_labels.append(self._y[index])

        # print('k_nearest_class_labels', k_nearest_class_labels)

        predicted_class = None

        if self.weights == 'uniform':
            # finding the most common labels, using uniform approach
            (unique_classes, counts) = np.unique(k_nearest_class_labels, return_counts=True)

            predicted_class = unique_classes[np.argmax(counts)]

        else:
            class_weights_map = {}
            dist_lcm = np.prod(np.take(cal_dist_arr, k_nearest_indices))

            for i in range(len(k_nearest_class_labels)):
                class_label = k_nearest_class_labels[i]

                if class_label not in class_weights_map:
                    class_weights_map[class_label] = 0

                class_weights_map[class_label] += dist_lcm/cal_dist_arr[k_nearest_indices[i]]

            max_weight = None
            for (each_class, weight) in class_weights_map.items():
                if not max_weight:
                    max_weight = weight
                    predicted_class = each_class
                elif weight > max_weight:
                    max_weight = weight
                    predicted_class = each_class

        # print('predicted_class', predicted_class)

        return predicted_class

