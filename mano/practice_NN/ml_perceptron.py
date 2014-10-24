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

        ### TODO ###

        # 1, Set activation function on hidden layer i.e. self.actHid
        # choices are tabh or sigmoid. self.d_actHid should be set as well



        # 2, Set activation function on output layer
        # choices are tabh or sigmoid or softmax. self.d_actOut does NOT need to set




        # 3, Randomly initialize weights between -1.0 to 1.0
        self.wHid =
        self.wOut =


    def train(self, X, t, eta=0.02, iteration=10000):
        """ Training weights
        Function Inputs:
        X (2d-array)    Train features (Number of Data * Feature Demension)
        t (array)       Train labels
        eta             Learning Ratio
        iteration       The number of updating time
        """
        # Add bias terms
        X = np.hstack([np.ones([X.shape[0], 1]), X])
        t = np.array(t)
        # Online learning
        for iter in range(iteration):
            if (iter+1)%1000 == 0: print iter+1,'/', iteration
            ### TODO ###
            # 1, ramdonly choose an index of data
            ind =
            # 2, set train feature (use the index you picked above)
            input =
            # 3, forward computation
            hidden_base =
            hidden_act =
            output_base =
            output_act =
            # 4, cross entropy error function & Backward computation
            deltaOut =
            deltaHid =
            # 5, update weights
            self.wHid -=
            self.wOut -=

    def test(self, sample):
        """ Test perceptron
        Function Inputs:
        sample (array)    Test images feature
        """
        ### TODO ###
        # 1, add bias term
        input =
        # 2, forward computation
        hidden_act =
        output_act =
        return output_act

    def _tanh(self, x):
        return ### TODO ###

    def _d_tanh(self, x):
        return ### TODO ###

    def _sigmoid(self, x):
        return ### TODO ###

    def _d_sigmoid(self, x):
        return ### TODO ###

    def _softmax(self, x):
        return ### TODO ###


if __name__ == "__main__":
    # sample XOR
    model = MultiLayerPerceptron(2, 5, 1, "tanh", "sigmoid")
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    t = np.array([0, 1, 1, 0])
    # train & test
    model.train(X,t)
    res = [ model.test(sample) for sample in X ]
    # show
    print "Input", "Score", "Label"
    for ind, item in enumerate(zip(res, t)):
        print X[ind], item
