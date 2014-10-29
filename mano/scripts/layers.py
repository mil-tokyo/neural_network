# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Layer Data Strucrue """

import numpy as np

class Layer(object):
    """ base layer class """
    def __init__(self, name='', prev=[], nex=[]):
        self.name, self.prev, self.next = name, prev, next

    # method for fc and conv
    def initialize_weigths(self, elements):
        layerPair = [bottom+'_'+top for bottom in self.prev for top in self.next]
        self.w = {}
        for pair in layerPair:
            self.[pair] = np.random.uniform(-1.0, 1.0, elements)
    def get_weights(self, bottom, top):
        return self.w[bottom+'_'+top]
    def set_weights(self, bottom, top, value):
        self.w[bottom+'_'+top] = value
    def save_weights(self, path='', bottom, top):
        name = bottom+'_'+top
        np.save(path+name, self.w[name])

    # method for conv and pool (the most left-top filter will be arranged on edges)
    def compute_filter_centers(self, imageSize):
        baseY, baseX = (self.patch[0]-1)/2, (self.patch[1]-1)/2
        centerListY = xrange(baseY,imageSize[0],self.stride)
        centerListX = xrange(baseX,imageSize[1],self.stride)
        new_imageSize = len(centerListY), len(centerListX)
        centers = [ (h, w, i, j) for i,h in enumerate(centerListY) for j,w in emumerate(centerListX) ]
        return new_imageSize, center

    # get funcs
    def get_name(self):
        return self.name
    def get_prev(self):
        return self.prev
    def get_next(self):
        return self.next


class FullyConnected(Layer):
    """ fc layer class """
    def __init__(self, name, prev=[], next=[], elements=None, drop_ratio=.0):
        super(FullyConnected, self).__init__(name, prev, next)
        super(FullyConnected, self).initialize_weigths(elements)
        self.drop_ratio = drop_ratio
    # compute filter reaction
    def filter(self, bottom, top, valueInput):
        featureDim = valueInput.shape[1]*valueInput.shape[2]*valueInput.shape[3]
        valueOutput = np.zeros(shape(valueInput.shape[0], featureDim, 1, 1))
        weight = self.get_weigths(bottom, top)
        for batch in valueInput.shape[0]:
            feature = valueInput[batch,:,:,:].reshape( featureDim )
            valueOutput[batch,:,:,:] = (weight * feature).reshape((featureDim, 1, 1))
        return valueOutput
    # choose drop out
    def choose_drop_incide(self):
        return np.random.random_integers(0, self.channel, int(self.channel*self.dropratio))
    # get funcs
    def get_drop_ratio(self):
        return self.drop_ratio


class Convolution(Layer):
    """ conv layer class """
    def __init__(self, name='', prev=[], next=[], elements=None, stride=2)
        super(Convolution, self).__init__(name, prev, next)
        super(Convolution, self).initialize_weigths(elements)
        self.stride = stride
        self.patch = elements.shape[2:]
    # compute filter reaction
    def filter(self, valueInput):
        pass
    # get funcs
    def get_stride(self):
        return self.stride


class Activation(Layer):
    """ act layer class """
    def __init__(self, name='', prev=[], next=[], actType='tanh')
        super(Activation, self).__init__(name, prev, next)
        self.func = getattr(self, actType)
        self.d_func = getattr(self, 'd_'+actType) if actType != 'softmax'
    # compute activation
    def filter(self, valueInput):
        return self.func(valueInput)
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
    def relu(self.x):
        return np.maximum(0.0, x)
    def d_relu(x):
        res = np.zeros(x.shape)
        res[x >= 0.0] = 1.0
        return res
    # only for the last fc layer
    def softmax(self, x):
        return np.exp(x) / np.sum( np.exp(x) )


class Pooling(Layer):
    """ pool layaer class """
    def __init__(self, name='', prev=[], next=[], patch=(5,5), stride=2, poolType='max')
        super(Pooling, self).__init__(name, prev, next)
        self.patch = patch
        self.stride = stride
        self.poolType = poolType
    # compute pooling (max and mean)
    def filter(self, valueInput):
        new_imageSize, centers = self.compute_filter_centers(valueInput.shape[2:])
        valueOutput = np.zeros(shape=(valueInput.shape[0], valueInput.shape[1], new_imageSize[0], new_imageSize[1]))
        areaY, areaX = (self.patch[0]-1)/2, (self.patch[1]-1)/2
        for batch in xrange(valueInput.shape[0]):
            for channel in xrange(valueInput.shape[1]):
                im = valueInput[batch, channel,:,:]
                for centerY, centerX, newIndexY, newIndexX in centers:
                    cropedImage = im[ max(0, centerY-areaY):centerY+areaY, max(0, centerX-areaX):centerX+areaX ]
                    pool = getattr(cropedImage, poolType)
                    valueOutput[batch, channel, newIndexY, newIndexX] = pool()
        return valueOutput
