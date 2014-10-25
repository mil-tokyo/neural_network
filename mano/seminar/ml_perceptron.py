# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Multi Layer Perceptron
Three Layer: input-hidden-output """

import numpy as np
import sys

class MultiLayerPerceptron:
    def __init__(self, numInput, numHidden, numOutput, actHidden="tanh", actOutput="sigmoid"):
        """
        Function Inputs:
        numInput    unit number in input layer (without bias unit)
        numHidden   unit number in hidden layer (without bias unit)
        numOutput   unit number in output layer
        actHidden   activation function on hidden layer (tanh or sigmoid)
        actOutput   activation function on output layer (tanh, sigmoid, identity, softmax)
        """
        # Add bias unit
        self.numIn = numInput + 1
        self.numHid = numHidden + 1
        self.numOut = numOutput

        # Set activation function on hidden layer
        if actHidden == "tanh":
            self.actHid = self._tanh
            self.d_actHid = self._d_tanh
        elif actHidden == "sigmoid":
            self.actHid = self._sigmoid
            self.d_actHid = self._d_sigmoid
        else:
            print "NameError: Use tanh or sigmoid for actHidden"
            sys.exit()

        # Set activation function on output layer
        if actOutput == "tanh":
            self.actOut = self._tanh
        elif actOutput == "sigmoid":
            self.actOut = self._sigmoid
        elif actOutput == "softmax":
            self.actOut = self._softmax
        else:
            print "NameError: Use tanh or sigmoid for actOutput"
            sys.exit()

        # Randomly initialize weights between -1.0 to 1.0
        self.wHid = np.random.uniform(-1.0, 1.0, (numHidden+1, numInput+1))
        self.wOut = np.random.uniform(-1.0, 1.0, (numOutput, numHidden+1))


    def train(self, X, t, eta=0.02, iteration=10000):
        """ Training weights
        Function Inputs:
        X (2d-array)    Train images features (Number of Data * Feature Demension)
        t (array)       Train images labels
        eta             Learning Ratio
        iteration       The number of updating time
        """
        # Add bias terms
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        t = np.array(t)
        # Online learning
        for iter in range(iteration):
            if (iter+1)%1000 == 0: print iter+1,'/', iteration
            ind = np.random.randint(X.shape[0])
            # Randomly choose train data
            input = X[ind]
            # Forward computation
            hidden_base = np.dot(self.wHid, input)
            hidden_act = self.actHid( hidden_base )
            output_base = np.dot(self.wOut, hidden_act)
            output_act = self.actOut( output_base )
            # Cross entropy error function & Backward computation
            deltaOut = output_act - t[ind]
            deltaHid = self.d_actHid(hidden_base) * np.dot(self.wOut.T, deltaOut)
            # Update weights
            self.wHid -= eta * np.dot( np.atleast_2d(deltaHid).T, np.atleast_2d(input) )
            self.wOut -= eta * np.dot( np.atleast_2d(deltaOut).T, np.atleast_2d(hidden_act) )

    def test(self, sample):
        """ Test perceptron
        Function Inputs:
        sample (array)    Test images feature
        """
        input = np.append(1, sample)
        hidden_act = self.actHid(np.dot(self.wHid, input))
        output_act = self.actOut(np.dot(self.wOut, hidden_act))
        return output_act

    def _tanh(self, x):
        return np.tanh(x)

    def _d_tanh(self, x):
        return 1.0 - self._tanh(x) ** 2

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _d_sigmoid(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def _softmax(self, x):
        return np.exp(x) / np.sum( np.exp(x) )


if __name__ == "__main__":
    # sample XOR
    model = MultiLayerPerceptron(2, 5, 1, "tanh", "sigmoid")
    X = list([[0,0], [0,1], [1,0], [1,1]])
    t = list([0, 1, 1, 0])
    # train & test
    model.train( np.array(X), np.array(t) )
    res = [ model.test(np.array(sample)) for sample in X ]
    # show
    print "Input", "Score", "Label"
    for ind, item in enumerate(zip(res, t)):
        print X[ind], item
