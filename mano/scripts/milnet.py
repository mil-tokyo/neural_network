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
        self.iteration = 50 if mode=='train' else 10
        self.learningRatio = 0.05
        self.batchSize = 10 if mode=='train' else 100
        self.lossFunc = 'mean_squared_error'
        self.printTiming = 10 if mode=='train' else 2
        self.weightSavePath = '/Users/manopin/data/tmp/'

    def layer_setting(self):
        """
        # 3 layer NN
        self.layers = [
            FC(name='fc1', prev=['input'], next=['act1'], elements=(np.prod(self.inputSize), 100), dropRatio=.0),
            Act(name='act1', prev=['fc1'], next=['fc2'], actType='tanh'),
            FC(name='fc2', prev=['act1'], next=['act2'], elements=(100, self.classNum), dropRatio=.0),
            Act(name='act2', prev=['fc2'], next=['output'], actType='softmax')
            ]
        """

        # 4 layer conv
        self.layers = [
            Conv(name='conv1', prev=['input'], next=['pool1'], elements=(10, 1, 5, 5), stride=1),
            Pool(name='pool1', prev=['conv1'], next=['act1'], patch=(2,2), stride=2, poolType='max', flagLinkFC=False),
            Act(name='act1', prev=['pool1'], next=['conv2'], actType='tanh'),
            Conv(name='conv2', prev=['act1'], next=['pool2'], elements=(12, 10, 3, 3), stride=1),
            Pool(name='pool2', prev=['conv2'], next=['act2'], patch=(2,2), stride=2, poolType='max', flagLinkFC=True),
            Act(name='act2', prev=['pool2'], next=['fc1'], actType='tanh'),
            FC(name='fc1', prev=['act2'], next=['act3'], elements=(300, 128), dropRatio=.0),
            Act(name='act3', prev=['fc1'], next=['fc2'], actType='tanh'),
            FC(name='fc2', prev=['act3'], next=['act4'], elements=(128, 10), dropRatio=.0),
            Act(name='act4', prev=['fc2'], next=['output'], actType='softmax')
            ]
