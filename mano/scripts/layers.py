# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Layer Data Strucrue """

import numpy as np

class Layer(object):
    """ base layer class """
    def __init__(self, name='', prev=[], next=[]):
        self.name, self.prev, self.next = name, prev, next

    # store input data and propageted delta error
    def set_input_data(self, inputData):
        self.inputData = inputData
    def get_input_data(self):
        return self.inputData

    # method for fc and conv
    def initialize_weigths(self, elements):
        layerPair = [bottom+'_'+top for bottom in self.prev for top in self.next]
        self.w = {}
        for pair in layerPair:
            self.w[pair] = np.random.uniform(-1.0, 1.0, elements)
    def get_weights(self, bottom, top):
        return self.w[bottom+'_'+top]
    def set_weights(self, bottom, top, value):
        self.w[bottom+'_'+top] = value
    def save_weights(self, path, bottom, top):
        name = bottom+'_'+top
        np.save(path+name, self.w[name])

    def set_proped_delta(self, propedDelta):
        self.propedDelta = propedDelta
    def get_proped_delta(self):
        return self.propedDelta

    # method for conv and pool (the most left-top filter will be arranged on edges)
    def compute_filter_centers(self, imageSize):
        baseY, baseX = (self.patch[0]-1)/2, (self.patch[1]-1)/2
        centerListY = xrange(baseY,imageSize[0],self.stride)
        centerListX = xrange(baseX,imageSize[1],self.stride)
        newImageSize = len(centerListY), len(centerListX)
        centers = [ (h, w, i, j) for i,h in enumerate(centerListY) for j,w in emumerate(centerListX) ]
        return newImageSize, center

    # get funcs
    def get_name(self):
        return self.name
    def get_prev(self):
        return self.prev
    def get_next(self):
        return self.next
    def get_both(self):
        return self.prev, self.next


class FC(Layer):
    """ Fully Connected layer class """
    def __init__(self, name, prev=[], next=[], elements=None, dropRatio=.0):
        super(FC, self).__init__(name, prev, next)
        super(FC, self).initialize_weigths(elements)
        self.dropRatio = dropRatio
    # compute filter reaction
    def filter(self, bottom, top):
        weight = super(FC, self).get_weights(bottom, top)
        input = super(FC, self).get_input_data()
        return np.dot(input, weight)
    # compute filter back reaction
    def filter_back(self, bottom, top, deAct):
        weight = super(FC, self).get_weights(bottom, top)
        delta = super(FC, self).get_proped_delta()
        return np.dot(delta, weight.T) * deAct
    # update
    def update(self, bottom, top, learningRatio):
        weight = super(FC, self).get_weights(bottom, top)
        input = super(FC, self).get_input_data()
        delta = super(FC, self).get_proped_delta()
        new_weight = weight - learningRatio * np.dot(input.T, delta)
        print 'in', input[1]
        print 'delta', delta[1]
        print 'bottom', 'w', weight[1]
        print 'bottom', 'nw', new_weight[1]
        super(FC, self).set_weights(bottom, top, new_weight)
    # choose drop out
    ## TODO
    def choose_drop_incide(self, bottom, top):
        return np.random.random_integers(0, self.channel, int(self.channel*self.dropRatio))


class Conv(Layer):
    """ Convolution layer class """
    def __init__(self, name='', prev=[], next=[], elements=None, stride=2):
        super(Conv, self).__init__(name, prev, next)
        super(Conv, self).initialize_weigths(elements)
        self.stride = stride
        self.patch = elements.shape[2:]
    # compute filter reaction
    def filter(self, bottom, top):
        pass


class Act(Layer):
    """ Activation layer class """
    def __init__(self, name='', prev=[], next=[], actType='tanh'):
        super(Act, self).__init__(name, prev, next)
        self.func = getattr(self, actType)
        if actType != 'softmax': self.d_func = getattr(self, 'd_'+actType)
    # compute filter reaction
    def filter(self, bottom, top):
        return self.func( super(Act, self).get_input_data() )
    # compute filter back reaction
    def filter_back(self):
        return self.d_func( super(Act, self).get_input_data() )
    # update
    def update(self, bottom, top, learningRatio):
        pass
    # functions
    def tanh(self, x):
        return np.tanh(x)
    def d_tanh(self, x):
        return 1.0 - self.tanh(x) ** 2
    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    def d_sigmoid(self, x):
        x = self.sigmoid(x)
        return x * (1.0 - x)
    def identity(self, x):
        return x
    def d_identity(self, x):
        return np.ones(shape=(x.shape))
    def relu(self, x):
        return np.maximum(0.0, x)
    def d_relu(self, x):
        res = np.zeros(x.shape)
        res[x >= 0.0] = 1.0
        return res
    # only for the last fc layer
    def softmax(self, x):
        return np.exp(x) / np.sum( np.exp(x) )


class Pool(Layer):
    """ Pooling layaer class """
    def __init__(self, name='', prev=[], next=[], patch=(5,5), stride=2, poolType='max'):
        super(Pool, self).__init__(name, prev, next)
        self.patch = patch
        self.stride = stride
        self.poolType = poolType
    # compute filter reaction (max and mean)
    def filter(self, bottom, top):
        valueInput = super(Pool, self).get_input_data()
        new_imageSize, centers = super(Pool, self).compute_filter_centers(valueInput.shape[2:])
        if new_imageSize == (1,1): # if this links to a fc layer
            valueOutput = np.zeros(shape=valueInput.shape[:2])
        else:
            valueOutput = np.zeros(shape=(valueInput.shape[0], valueInput.shape[1], new_imageSize[0], new_imageSize[1]))
        areaY, areaX = (self.patch[0]-1)/2, (self.patch[1]-1)/2
        for batch in xrange(valueInput.shape[0]):
            for channel in xrange(valueInput.shape[1]):
                im = valueInput[batch, channel,:,:]
                for centerY, centerX, newIndexY, newIndexX in centers:
                    cropedImage = im[ max(0, centerY-areaY):min(centerY+areaY, valueInput.shape[2]), max(0, centerX-areaX):min(centerX+areaX, valueInput.shape[3]) ]
                    pool = getattr(cropedImage, poolType)
                    if new_imageSize == (1,1): # if this links to a fc layer
                        valueOutput[batch, channel] = pool()
                    else:
                        valueOutput[batch, channel, newIndexY, newIndexX] = pool()
        return valueOutput
