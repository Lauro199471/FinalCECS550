import numpy as np
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from random import randrange


class NeuralNetwork(BaseEstimator):
    """
    Represents a neural network.
    """

    def __init__(self, lr=None, epoch=None):
        self.lr = lr
        self.epoch = epoch
        self._layers = []

    def add_layer(self, layer):
        """
        Adds a layer to the neural network.
        :param Layer layer: The layer to add.
        """
        self._layers.append(layer)

    def feed_forward(self, X):
        """
        Feed forward the input through the layers.
        :param X: The input values.
        :return: The result.
        """
        X = np.reshape(X, (1, -1))  # make sure input is 1xn
        #print("====================\n=== FEED FORWARD ====\n====================")
        #print("X: ", type(X), np.shape(X))
        # print("\n\n")
        i = 0
        for layer in self._layers:
            # print("i: ", i)
            X = layer.activate(X)
            # print("X: ", type(X), np.shape(X))
            i += 1
            # print("\n\n")
        return X

    def predict(self, X):
        """
        Predicts a class (or classes).
        :param X: The input values.
        :return: The predictions.
        """
        X = np.reshape(X, (1, -1))
        predication = self.feed_forward(X)
        # predication = np.argmax(A, axis=0)
        return predication

    def backpropagation(self, X, y):
        """
        Performs the backward propagation algorithm and updates the layers weights.
        :param X: The input values.
        :param y: The target values.
        """
        X = np.reshape(X, (1, -1))
        y = np.reshape(y, (1, -1))
        # print("x:", X, " shape: ", X.shape)
        j = 0
        # Feed forward for the output
        output = self.feed_forward(X)
        # Loop over the layers backward
        # print("========================================\n=== BACK PROPAGATION "
        #       "====\n========================================")
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            # If this is the output layer
            if layer == self._layers[-1]:
                # print("j: ", j)
                j += 1

                prev_layer = self._layers[i - 1]
                '''
                δError/δo = prediction - target (using mean square error)
                '''
                layer.error = output - y
                # print("layer.error:(δE/δo)\n", layer.error, " ", layer.error.shape)
                '''
                δo/δzo = activation derivative(zo)
                '''
                do_dz = layer.apply_activation_derivative(output)
                # print("do_dz:(δo/δzo)\n", do_dz, " ", do_dz.shape)
                '''
                δz/δw = input(previous layer after activation from previous layer)
                '''
                dz_dw = prev_layer.last_activation
                # print("dz_dw:(δz/δw)\n", dz_dw, " ", dz_dw.shape)
                '''
                δerror_total/δw(i) = δerror_total/δout(i) * δout(i)/δnet * δnet/δw(i)
                '''
                layer.delta = np.dot((layer.error * do_dz).T, dz_dw)
                # print("layer.delta:(δerror/δw)\n", layer.delta, " ", layer.delta.shape)

            else:
                # print("j: ", j)
                j += 1

                next_layer = self._layers[i + 1]
                '''
                δerror_total/δh = δerror_total/δout(i+1) * δout(i+1)/δnet
                '''
                de_do = next_layer.error
                # print("de_do:(δe/δo)\n", de_do, " ", de_do.shape)
                do_dz = layer.apply_activation_derivative(next_layer.last_activation)
                # print("do_dz:(δo/δz)\n", do_dz, " ", do_dz.shape)
                dz_dh = next_layer.weights
                # print("dz_dh:(δz/δh)\n", dz_dh, " ", dz_dh.shape)
                layer.error = np.dot(de_do * do_dz, dz_dh.T)
                # print("layer.error: \n", layer.error)
                '''
                δh/δzh = activation derivative(zo)
                '''
                dz_dhh = layer.apply_activation_derivative(layer.last_activation)
                # print("dz_dhh:(δz/δh)\n", dz_dhh, " ", dz_dhh.shape)
                '''
                δde/δw = δerror_total/δh * δh/δzh * δzh/δw
                '''
                layer.delta = np.dot((layer.error * dz_dhh).T, layer.input)
                # print("layer.delta:\n", layer.delta, " ", layer.delta.shape)

        # print("\nupdating...")
        # Update the weights
        for i in range(len(self._layers)):
            layer = self._layers[i]
            deltas = np.reshape(layer.delta, (1, -1))
            # print("deltas: \n", deltas)
            weights = np.reshape(layer.weights, (1, -1))
            # print("weights: \n", weights)
            layer.updateweights(weights - (self.lr * deltas))
            # print("weights: \n", layer.weights)
            # print("\n\n")

    def fit(self, Xtrain, ytrain):
        """
        Trains the neural network using backpropagation.
        :param ytrain:
        :param Xtrain:
        """
        print("training...")
        for i in range(self.epoch):
            if i == np.round(self.epoch / 4) and i != 0:
                print("25% done....")
            if i == np.round(self.epoch / 2) and i != 0:
                print("50% done....")
            if i == np.round((3 * self.epoch) / 4) and i != 0:
                print("75% done....")

            for j in range(len(Xtrain)):
                self.backpropagation(Xtrain[j], ytrain[j])
        print("done...")

        # if i % 100 == 0:
        #    c = 0
        #    for test_i in range(len(Xtest)):
        #        a = ytest[test_i] - self.feed_forward(Xtest[test_i])
        #        b = np.power(a, 2)
        #        c = c + b

        #    mse = c / len(Xtest)  # np.mean(np.square(y[2] - self.feed_forward(X[2])))
        #    mses.append(mse)
        # print(type(mse), np.shape(mse))
        #    print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        # mses.append(mse)

        # for h in range(len(X)):
        #    print(self.feed_forward(X[h]))
        #    mse = np.mean(np.square(y[i] - self.feed_forward(X[i])))
        # print(predictions)
        # print(ytest)
        # mse = mean_squared_error(ytest, predictions)
        # print("Here")
        # print(type(y), np.shape(y))
        # print(type(predictions), np.shape(predictions))
        # mse = mean_squared_error(y, predictions)
        # return mses

    # Mean Squared Error
    def score(self, X, Y):
        predicitons_par = []
        for i in range(len(X)):
            a = self.feed_forward(X[i, :])
            predicitons_par.append(a[0])

        c = 0
        for (pred, real) in zip(predicitons_par, Y):
            a = pred - real
            b = np.power(a, 2)
            c = c + b
        mse = c / len(X)
        print("mse: ", mse)
        return mse
