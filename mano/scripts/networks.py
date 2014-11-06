# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Network Data structure """

from layers import *
from time import time

# TODO: deal with branching computation
class Network:

    def __init__(self, layers, lossFunction='crorss_entropy'):
        self.layer = dict([ (layerConfig.get_name(), layerConfig) for layerConfig in layers])
        self.firstLayerNames = [name for name, layerConfig in self.layer.iteritems() if 'input' in layerConfig.get_prev()]
        self.lastLayerNames = [name for name, layerConfig in self.layer.iteritems() if 'output' in layerConfig.get_next()]
        self.lossFunc = getattr(self, lossFunction)
        self.d_lossFunc = getattr(self, 'd_'+lossFunction)

    # save layer weights
    def save_weights(self, path):
        for name, layerInst in self.layer.iteritems():
            if 'act' in name or 'pool' in name: continue
            bottom, top = layerInst.get_both()
            layerInst.save_weights(path, bottom[0], top[0])

    # load layer weights
    def load_weights(self, path):
        for name, layerInst in self.layer.iteritems():
            if 'act' in name or 'pool' in name: continue
            bottom, top = layerInst.get_both()
            layerInst.load_weights(path, bottom[0], top[0])

    # test network
    def test(self, datasetInst, batchSize, iteration, learningRatio, printTiming, path=None):
        if path is not None: self.load_weights(path)
        startTime, errorRatio = time(), 0
        for iter in xrange(iteration):
            input, target = datasetInst.create_batch_input(batchSize)
            output = self.forward_computation(input)
            res, ans = output.argmax(axis=1), target.argmax(axis=1)
            errorRatio += len(np.nonzero(res-ans)[0])
            if (iter+1)%printTiming==0: print 'Time:%f' % (time()-startTime), 'Testing:%d/%d(batch%d)' % (iter+1, iteration, batchSize), 'Loss:%f' % self.lossFunc(output, target)
        print 'Time:%f' % (time()-startTime), 'ErrorRatio:%f' % (errorRatio/float(iteration*batchSize))

    # train network
    def train(self, datasetInst, batchSize, iteration, learningRatio, printTiming, path=None):
        startTime = time()
        for iter in xrange(iteration):
            input, target = datasetInst.create_batch_input(batchSize)
            output = self.forward_computation(input)
            error = self.d_lossFunc(output, target)
            self.backward_computation(error)
            self.update(learningRatio)
            if (iter+1)%printTiming==0: print 'Time:%f' % (time()-startTime), 'Training:%d/%d(batch%d)' % (iter+1, iteration, batchSize), 'Loss:%f' % self.lossFunc(output, target)
        if path is not None: self.save_weights(path)

    # forward propergation
    def forward_computation(self, valueInput):
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
            if 'act' in currentName: currentName = self.layer[currentName].get_prev()[0]; continue # if act, skip bp computation
            self.layer[currentName].set_proped_delta(currentDelta)
            bottom, top = self.layer[currentName].get_both()
            if 'input' in bottom[0]: break # no need to bp if the next layer is input
            deAct = self.layer[bottom[0]].filter_back() if 'pool' not in currentName else None # no need to set deAct if this layer is pooling layer
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
        return np.square(prediction - target).sum() / 2.0 / prediction.shape[0]
    def d_mean_squared_error(self, prediction, target):
        return prediction - target
    def cross_entropy(self, prediction, target):
        return -target * np.log(prediction) - (1.0-target) * np.log(1.0-prediction)
    def d_cross_entropy(self, prediction, target):
        return prediction - target
