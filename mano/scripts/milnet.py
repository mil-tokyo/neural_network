# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Conigure file for milnet """

from dataset import *
from layers import *

class Setting(object):

    def __init__(self, mode):

        # dataset
        self.dataset_setting(mode)
        # parameters
        self.basic_setting(mode)
        # layers
        self.layer_setting()
        self.layers = self.create_layer_dictionary()

    def dataset_setting(self, mode):
        self.datasetName = 'MNIST'
        self.ImageDir = '/Users/manopin/data/mnist/'+mode
        self.LabelDir = '/Users/manopin/data/mnist/%s-labels.txt' % mode
        self.inputSize = None # set this if you want to resize input images
        self.datasetInst = eval(self.datasetName)(self.ImageDir, self.LabelDir, self.inputSize)

    def basic_setting(self, mode):
        self.iteration = 100000
        self.batchSize = 128 if mode=='train' else 500

    def layer_setting(self):
        """ Examples
        cf. data = np.array(shape(batchSize, channels, heigth, width))
        fc1 = self.set_params(layeyType='FullyConnected', name='fc1', prev=['input'], next=['act1'], elements=(1024, 1024), dropout=.0) # input_layer, output_layer
        act1 = self.set_params(layerType='Activation', name='act1', prev=['fc1'], next=['fc2'], actType='tanh') # tanh, sigmoid, relu, identity, softmax
        conv1 = self.set_params(layerType='Convolution', name='conv1', prev=[None], next=['act1'], elements=(128, 192, 7, 7), stride=2) # channel_in, channel_out, row, col
        pool1 = self.set_params(layerType='Pooling', name='act1', prev=['conv1'], next=['conv2'], patch=(5,5), stride=2, poolType='max') # max, mean """

        return layers = [
            FullyConnected(name='fc1', prev=['input'], next=['act1'], elements=(self.datasetInst.inputRow*self.datasetInst.inputCol, 100), dropout=.0),
            Activation(name='act1', prev=['fc1'], next=['fc2'], actType='tanh'),
            FullyConnected(name='fc2', prev=['act1'], next=['act2'], elements=(100, 10), dropout=.0),
            Activation(name='act2', prev=['fc2'], next=['output'], actType='softmax')
            ]

    for create_layer_dictionary(self):
        return dict([ (layer.name, layer) for layer in layers])
