import numpy as np
from scipy.special import expit, softmax
from sklearn.base import BaseEstimator
import tensorflow as tf

class Layer(BaseEstimator):
    """
     Represents a layer (hidden or output) in our neural network.
     """

    def __init__(self, n_input, n_neurons, activation=None, w=None, b=None):
        """
         :param int n_input: The input size (coming from the input layer or a previous hidden layer)
         :param int n_neurons: The number of neurons in this layer.
         :param str activation: The activation function to use (if any).
         :param weights: The layer's weights.
         :param bias: The layer's bias.
         """
        self.last_activation = None
        self.error = None
        self.delta = None
        self.input = None
        self.activation = activation
        self.n_input = n_input
        self.n_neurons = n_neurons
        '''
        NUMPY
        '''
        self.weights = np.random.rand(n_input, n_neurons)
        self.bias = np.random.random_sample(n_neurons)
        '''
        TENSORFLOW
        '''
        self.weights_tf = tf.constant(self.weights)
        self.bias_tf = tf.constant(self.bias)


    def updateweights(self, w):
        self.weights = np.reshape(w, (self.n_input, self.n_neurons))

    def activate(self, x):
        """
        Calculates the dot product of this layer.
        :param x: The input.
        :return: The result.
        """
        self.input = x
        # print("x:\n", x, " ", x.shape)
        # print("weights:\n", self.weights, " ", self.weights.shape)
        # print("bias:\n", self.bias, " ", self.bias.shape)

        r = np.dot(x, self.weights) + self.bias

        # print("h:\n", r, " ", r.shape)

        self.last_activation = self._apply_activation(r)
        #print("zh:\n", self.last_activation, " ", self.last_activation.shape)

        return self.last_activation

    def _apply_activation(self, r):
        """
        Applies the chosen activation function (if any).
        :param r: The normal value.
        :return: The activated value.
        """
        if self.activation is None:
            """
            No activation function was chosen.
            """
            return r
        if self.activation == 'tanh':
            """
            Compute tanh values for each sets of scores in x.
            """
            return np.tanh(r)
        if self.activation == 'sigmoid':
            """
            Compute sigmoid values for each sets of scores in x.
            """
            return expit(r)  # 1 / (1 + np.exp(-r))
        if self.activation == 'softmax':
            """
            Compute softmax values for each sets of scores in x.
            """
            # exps = np.exp(r - np.max(r))
            return softmax(r)  # exps / np.sum(exps)
        return r

    def apply_activation_derivative(self, r):
        """
        Applies the derivative of the activation function (if any).
        :param r: The normal value.
        :return: The "derived" value.
        """
        # We use 'r' directly here because its already activated, the only values that
        # are used in this function are the last activations that were saved.
        if self.activation is None:
            return r
        if self.activation == 'tanh':
            return 1 - r ** 2
        if self.activation == 'sigmoid':
            return r * (1 - r)
        if self.activation == 'softmax':
            return r * (1 - r)
        return r
