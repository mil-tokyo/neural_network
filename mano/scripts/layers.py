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
    # weight
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
    def load_weights(self, path, bottom, top):
        name = bottom+'_'+top
        self.w[name] = np.load(path+name+'.npy')
    # delta
    def set_proped_delta(self, propedDelta):
        self.propedDelta = propedDelta
    def get_proped_delta(self):
        return self.propedDelta

    # method for conv and pool
    # get place ot be filtered (the most left-top filter will be arranged on edges)
    def compute_filter_place(self, imageSize):
        if hasattr(self, 'newImageSize'): return self.newImageSize, self.fixedOldImageSize, self.coordinates
        baseYs = range(0, imageSize[0]-self.patch[0]+1, self.stride)
        baseXs = range(0, imageSize[1]-self.patch[1]+1, self.stride)
        newImageSize = len(baseYs), len(baseXs)
        fixedOldImageSize = (baseYs[-1]+self.patch[0], baseXs[-1]+self.patch[1])
        coordinates = [ (h, w, h + self.patch[0], w + self.patch[1]) for h in baseYs for w in baseXs ]
        self.newImageSize, self.fixedOldImageSize, self.coordinates = newImageSize, fixedOldImageSize, coordinates
        return newImageSize, fixedOldImageSize, coordinates
    # get configs
    def get_filter_config(self, weightData=None, inputData=None):
        # once it has stored, just return the value
        if hasattr(self, 'bSize'): return self.bSize, self.cInSize, self.cOutSize, self.oriImInSize, self.fixedImInSize, self.oriImOutSize, self.fixedImOutSize, self.coord
        # compute value in the first iter
        self.bSize, self.cInSize, height, width = inputData.shape
        self.cOutSize = weightData.shape[0] if weightData is not None else None
        self.oriImInSize = (height, width)
        self.oriImOutSize, self.fixedImInSize, self.coord = self.compute_filter_place(self.oriImInSize)
        self.fixedImOutSize = None if weightData is not None else np.prod(self.oriImOutSize) * self.cInSize
        return self.bSize, self.cInSize, self.cOutSize, self.oriImInSize, self.fixedImInSize, self.oriImOutSize, self.fixedImOutSize, self.coord

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
        input = super(FC, self).get_input_data()
        delta = super(FC, self).get_proped_delta()
        return np.dot(delta, weight.T) * deAct( input )

    # update
    def update(self, bottom, top, learningRatio):
        weight = super(FC, self).get_weights(bottom, top)
        input = super(FC, self).get_input_data()
        delta = super(FC, self).get_proped_delta()
        weightNew = weight - learningRatio * np.dot(input.T, delta) / float(input.shape[0])
        super(FC, self).set_weights(bottom, top, weightNew)

    # choose drop out indice
    ## TODO
    def choose_drop_incide(self, bottom, top):
        return np.random.random_integers(0, self.channel, int(self.channel*self.dropRatio))


class Conv(Layer):
    """ Convolution layer class """
    def __init__(self, name='', prev=[], next=[], elements=None, stride=2):
        super(Conv, self).__init__(name, prev, next)
        super(Conv, self).initialize_weigths(elements)
        self.stride = stride
        self.patch = elements[2:]

    # compute filter reaction
    def filter(self, bottom, top):
        # setting
        weight = super(Conv, self).get_weights(bottom, top)
        input = super(Conv, self).get_input_data()
        if hasattr(self, 'bSize') and self.bSize != input.shape[0]: self.bSize = input.shape[0]
        batchSize, cInSize, cOutSize, oriImInSize, fixedImInSize, oriImOutSize, fixedImOutSize, coord = super(Conv, self).get_filter_config(weightData=weight, inputData=input)
        fixedInput = np.zeros(shape=(batchSize, cInSize, fixedImInSize[0], fixedImInSize[1]))
        fixedInput[:, :, :oriImInSize[0], :oriImInSize[1]] = input
        output = np.empty(shape=(batchSize, cOutSize, oriImOutSize[0], oriImOutSize[1]))
        # calc
        for batch in xrange(batchSize):
            im = fixedInput[batch, :, :, :]
            for outFilterInd in xrange(cOutSize):
                inFilter = weight[outFilterInd, :, :, :]
                filteredVal = [ np.sum( im[:, startY:endY, startX:endX] * inFilter ) for startY,startX,endY,endX in coord ]
                output[batch, outFilterInd, :, :] = np.array(filteredVal).reshape(oriImOutSize)
        return output

    # compute filter back reaction
    def filter_back(self, bottom, top, deAct):
        # setting
        weight = super(Conv, self).get_weights(bottom, top)
        input = super(Conv, self).get_input_data()
        delta = super(Conv, self).get_proped_delta()
        batchSize, cInSize, cOutSize, oriImInSize, fixedImInSize, oriImOutSize, fixedImOutSize, coord = super(Conv, self).get_filter_config(weightData=None, inputData=None)
        deFilteredTensor = np.empty(shape=(batchSize,cInSize,fixedImInSize[0],fixedImInSize[1])) ## output
        # calc
        for batch in xrange(batchSize):
            tmpDelta = delta[batch, :, :, :]
            for inFilterInd in xrange(cInSize):
                outFilter = weight[:, inFilterInd, :, :]
                tmpOutput = np.zeros(shape=fixedImInSize)
                deFilteredValList = [ (outFilter * tmpDelta[:,h,w].reshape(cOutSize,1,1)).sum(axis=0) for h in xrange(oriImOutSize[0]) for w in xrange(oriImOutSize[1]) ]
                for ind, deFilteredVal in enumerate(deFilteredValList):
                     startY,startX,endY,endX = coord[ind]
                     tmpOutput[startY:endY, startX:endX] += deFilteredVal
                deFilteredTensor[batch, inFilterInd, :, :] = tmpOutput
        output = deFilteredTensor[:, :, :oriImInSize[0], :oriImInSize[1]] * deAct( input )
        return output

    # update
    def update(self, bottom, top, learningRatio):
        # setting
        weight = super(Conv, self).get_weights(bottom, top)
        input = super(Conv, self).get_input_data()
        delta = super(Conv, self).get_proped_delta()
        batchSize, cInSize, cOutSize, oriImInSize, fixedImInSize, oriImOutSize, fixedImOutSize, coord = super(Conv, self).get_filter_config(weightData=None, inputData=None)
        updatePlace = self.get_update_place(oriImInSize, oriImOutSize)
        weightGrad = np.empty(shape=weight.shape)
        # calc
        for outFilterInd in xrange(cOutSize):
            tmpDelta = np.tile( delta[:,outFilterInd,:,:].reshape(batchSize,1,oriImOutSize[0],oriImOutSize[1]), (1,cInSize,1,1) )
            for deConvInd, (startY,startX,endY,endX) in enumerate(updatePlace):
                h = int(deConvInd/self.patch[1])
                w = deConvInd - h*self.patch[1]
                weightGrad[outFilterInd,:,h,w] = (input[:,:,startY:endY,startX:endX] * tmpDelta).sum(axis=3).sum(axis=2).sum(axis=0)
        weightNew = weight - learningRatio * weightGrad / float(batchSize)
        super(Conv, self).set_weights(bottom, top, weightNew)

    # compute update place
    def get_update_place(self, oriImInSize, oriImOutSize):
        if hasattr(self, 'updatePlace'): return self.updatePlace
        self.updatePlace = [ (h, w, h+oriImOutSize[0], w+oriImOutSize[1]) 
                 for h in xrange(0, oriImInSize[0]-oriImOutSize[0]+1, self.stride) for w in xrange(0, oriImInSize[1]-oriImOutSize[1]+1, self.stride) ]
        return self.updatePlace


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
        return self.d_func
    # update
    def update(self, bottom, top, learningRatio):
        pass

    # activation functions
    def tanh(self, x):
        return np.tanh(x)
    def d_tanh(self, x):
        return 1.0 - np.tanh(x) ** 2
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
        return np.sign(x)
    # only for the last fc layer
    def softmax(self, x):
        return (np.exp(x.T) / np.exp(x).sum(axis=1)).T


class Pool(Layer):
    """ Pooling layaer class """
    def __init__(self, name='', prev=[], next=[], patch=(5,5), stride=2, poolType='max', flagLinkFC=False):
        super(Pool, self).__init__(name, prev, next)
        self.patch = patch
        self.stride = stride
        self.poolType = poolType
        self.flagLinkFC = flagLinkFC

    # compute filter reaction (max and average)
    def filter(self, bottom, top):
        # setting
        input = super(Pool, self).get_input_data()
        if hasattr(self, 'bSize') and self.bSize != input.shape[0]: self.bSize = input.shape[0]
        batchSize, cInSize, cOutSize, oriImInSize, fixedImInSize, oriImOutSize, fixedImOutSize, coord = super(Pool, self).get_filter_config(weightData=None, inputData=input)
        output = np.empty(shape=(batchSize, cInSize, oriImOutSize[0], oriImOutSize[1]))
        backPropWeights = np.empty( shape=input.shape ) ## to preserve coordinates for back-prop
        # loop by each batch and channel
        for batch in xrange(batchSize):
            for channel in xrange(cInSize):
                # tmp val setting
                im = input[batch, channel,:,:] ## image to be filtered
                weight = np.zeros( shape=oriImInSize ) if self.poolType == 'max' else np.empty( shape=oriImInSize ) ## to store tmp weight
                tmpPooledValList = [] ## to store tmp pooled value
                # calc
                for startY,startX,endY,endX in coord:
                    cropedImage = im[ startY:endY, startX:endX ]
                    if self.poolType == 'max':
                        pooledValue = cropedImage.max()
                        pooledIndex = cropedImage.argmax()
                        tmpPooledValList.append( pooledValue )
                        h = int(pooledIndex/self.patch[1])
                        w = pooledIndex - h*self.patch[1]
                        weight[ startY+h, startX+w ] = 1.0
                    if self.poolType == 'average':
                        pooledvalue = cropedImage.mean()
                        tmpPooledValList.append( pooledValue )
                        weight[ startY:endY, startX:endX ] = cropedImage / pooledValue
                # store val
                output[batch, channel, :, :] = np.array( tmpPooledValList ).reshape( oriImOutSize )
                backPropWeights[batch, channel,:,:] = weight
        self.set_bp_weight( backPropWeights )
        if self.flagLinkFC: output = output.reshape( (batchSize, fixedImOutSize) ) ## resize if next is fc
        return output

    # compute filter back reaction
    def filter_back(self, bottom, top, deAct):
        # setting
        backPropWeights = self.get_bp_weight()
        delta = super(Pool, self).get_proped_delta()
        batchSize, cInSize, cOutSize, oriImInSize, fixedImInSize, oriImOutSize, fixedImOutSize, coord = super(Pool, self).get_filter_config(weightData=None, inputData=None)
        if delta.ndim == 2: delta = delta.reshape( (batchSize, cInSize, oriImOutSize[0], oriImOutSize[1]) ) ## if this layer connects to fc, reshape to original shape
        # calc by batch
        tmpDeltaReshaped = np.empty(shape=backPropWeights.shape)
        tmpDeltaPatchList = [ np.tile( delta[:,:,h,w].reshape((batchSize, cInSize, 1, 1)), (1, 1, self.stride, self.stride) )
                              for h in xrange(oriImOutSize[0]) for w in xrange(oriImOutSize[1]) ]
        for ind, (startY,startX,endY,endX) in enumerate(coord):
            tmpDeltaReshaped[:, :, startY:endY, startX:endX] = tmpDeltaPatchList[ind]
        return backPropWeights * tmpDeltaReshaped

    # update
    def update(self, bottom, top, learningRatio):
        pass

    # bp methods
    def set_bp_weight(self, backPropWeights):
        self.bpw = backPropWeights
    def get_bp_weight(self):
        return self.bpw
