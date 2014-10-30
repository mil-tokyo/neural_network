# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Network Data structure """

from layers import *
from tensors import Tensor

# TODO: deal with branching computation & conv, pool
class Network:

    def __init__(self, layers, lossFunction='crorss_entropy'):
        self.layer = dict([ (layerConfig.get_name(), layerConfig) for layerConfig in layers])
        self.firstLayerNames = [name for name, layerConfig in self.layer.iteritems() if 'input' in layerConfig.get_prev()]
        self.lastLayerNames = [name for name, layerConfig in self.layer.iteritems() if 'output' in layerConfig.get_next()]
        self.lossFunc = getattr(self, lossFunction)
        self.d_lossFunc = getattr(self, 'd_'+lossFunction)

    # test network
    def test(self, datasetInst, batchSize):
        input, target = datasetInst.create_batch_input(batchSize)
        output = self.forwad_computation(input)
        res, ans = output.argmax(axis=1), target.argmax(axis=1)
        print res[:20]
        print ans[:20]
        return len(np.nonzero(res-ans)[0]) / float(batchSize)

    # train network
    def train(self, datasetInst, batchSize, iteration, learningRatio):
        print self.layer['fc1'].get_weights('input','act1')
        for iter in xrange(iteration):
            input, target = datasetInst.create_batch_input(batchSize)
            output = self.forwad_computation(input)
            error = self.d_lossFunc(output, target)
            self.backward_computation(error)
            self.update(learningRatio)
            if (iter+1)%10 == 0: print 'Training:', iter+1, '/', iteration
        print self.layer['fc1'].get_weights('input','act1')

    # forward propergation
    def forwad_computation(self, valueInput):
        currentName = self.firstLayerNames[0]
        currentData = valueInput if 'fc' not in currentName else valueInput.reshape( (valueInput.shape[0], np.prod(valueInput.shape[1:])) )
        while 'output' not in currentName:
            self.layer[currentName].set_input_data(currentData)
            bottom, top = self.layer[currentName].get_both()
            currentData = self.layer[currentName].filter(bottom[0], top[0])
            currentName = self.layer[currentName].get_next()[0]
        return currentData

    # back propergation
    def backward_computation(self, error):
        currentName = self.lastLayerNames[0]
        currentDelta = error
        while 'input' not in currentName:
            if 'fc' in currentName:
                self.layer[currentName].set_proped_delta(currentDelta)
                bottom, top = self.layer[currentName].get_both()
                if 'input' in bottom[0]: break
                deAct = self.layer[bottom[0]].filter_back()
                currentDelta = self.layer[currentName].filter_back(bottom[0], top[0], deAct)
            currentName = self.layer[currentName].get_prev()[0]

    # update weights
    def update(self, learningRatio):
        currentName = self.firstLayerNames[0]
        while 'output' not in currentName:
            bottom, top = self.layer[currentName].get_both()
            self.layer[currentName].update(bottom[0], top[0], learningRatio)
            currentName = self.layer[currentName].get_next()[0]

    # loss functions
    def mean_squared_error(self, prediction, target):
        return np.square(prediction - target).sum() / 2.0
    def d_mean_squared_error(self, prediction, target):
        return prediction - target
    def cross_entropy(self, prediction, target):
        return -target * np.log(prediction) - (1.0-target) * np.log(1.0-prediction)
    def d_cross_entropy(self, prediction, target):
        return (prediction - target) / (prediction * (1.0 - prediction))
