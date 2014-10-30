# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Conigure file for milnet """

import numpy as np
from layers import *

class Setting(object):
    """ Network Settings """
    def __init__(self, mode):
        # dataset
        self.dataset_setting(mode)
        # parameters
        self.learning_setting(mode)
        # layers
        self.layer_setting()

    def get_dataset_setting(self):
        return self.imageDir, self.labelDir, self.inputSize, self.classNum, self.contrastNorm, self.zcaWhitening

    # Set parameters below for your neural network
    def dataset_setting(self, mode):
        self.datasetName = 'MNIST'
        self.imageDir = '/Users/manopin/data/mnist/'+mode
        self.labelDir = '/Users/manopin/data/mnist/%s-labels.txt' % mode
        self.inputSize = (28, 28) # set input image size (if values differ from default image size, images will be automatically resized)
        self.classNum = 10
        self.contrastNorm = True
        self.zcaWhitening= False

    def learning_setting(self, mode):
        self.iteration = 1
        self.learningRatio = 0.001
        self.batchSize = 128 if mode=='train' else 9999
        self.lossFunc = 'mean_squared_error'

    def layer_setting(self):
        """ Examples
        cf. data = np.array(shape(batchSize, channels, heigth, width))
        fc1 = self.set_params(layeyType='FullyConnected', name='fc1', prev=['input'], next=['act1'], elements=(1024, 1024), dropout=.0) # input_layer, output_layer
        act1 = self.set_params(layerType='Activation', name='act1', prev=['fc1'], next=['fc2'], actType='tanh') # tanh, sigmoid, relu, identity, softmax
        conv1 = self.set_params(layerType='Convolution', name='conv1', prev=[None], next=['act1'], elements=(128, 192, 7, 7), stride=2) # channel_in, channel_out, row, col
        pool1 = self.set_params(layerType='Pooling', name='act1', prev=['conv1'], next=['conv2'], patch=(5,5), stride=2, poolType='max') # max, mean """

        self.layers = [
            FC(name='fc1', prev=['input'], next=['act1'], elements=(np.prod(self.inputSize), 100), dropRatio=.0),
            Act(name='act1', prev=['fc1'], next=['fc2'], actType='tanh'),
            FC(name='fc2', prev=['act1'], next=['act2'], elements=(100, self.classNum), dropRatio=.0),
            Act(name='act2', prev=['fc2'], next=['output'], actType='softmax')
            ]

