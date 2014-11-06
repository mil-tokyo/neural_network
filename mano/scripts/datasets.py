# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Configure file for datasets """

import os
import numpy as np
from PIL import Image

class Dataset(object):
    """ base dataset class """
    def __init__(self):
        pass
    def create_batch_input(self, batchSize):
        # choose indice
        if self.batchInd + batchSize < self.dataNum:
            lastInd = self.batchInd + batchSize
            imageIndice = range(self.batchInd, lastInd)
        else:
            lastInd = self.batchInd + batchSize - self.dataNum
            imageIndice = range(self.batchInd, self.dataNum) + range(lastInd)
        self.batchInd = lastInd
        # create input tensor
        feat = np.empty(shape=(batchSize, self.inputChannel, self.inputRow, self.inputCol))
        target = np.zeros(shape=(batchSize, self.classNum))
        for batchIndex, imageIndex in enumerate(imageIndice):
            im = np.asarray(Image.open(self.dataPair[imageIndex][0]))
            newIm = np.empty(shape=(self.inputChannel, self.inputRow, self.inputCol))
            for color in xrange(self.inputChannel):
                newIm[color,:,:] = im[:,:,color] if self.flagColorChannel else im[:,:]
            feat[batchIndex,:,:,:] =  newIm
            target[batchIndex, self.dataPair[imageIndex][1]] = 1.0 ## 1-of-K representation
        if self.flagNorm: feat = self.global_contrast_normalization(feat) ## to be changed
        return feat, target
    # resize image
    def resize_image(self, image): ## TODO
        pass
    # gcn by each image patch
    def global_contrast_normalization(self, feat):
        for batch in xrange(feat.shape[0]):
            for channel in xrange(feat.shape[1]):
                im = feat[batch, channel, :, :]
                feat[batch, channel, :, :] = (im-im.mean()) / np.sqrt(im.var()+10)
        return feat
    # zca whitening
    def zca_whitening(self): ## TODO
        pass


class MNIST(Dataset):
    """ MNIST dataset class """
    def __init__(self, setting):
        imageDir, labelDir, inputSize, classNum, flagNorm, flagWhite = setting
        # make data pair
        paths = [os.path.join(root,file) for (root,dirs,files) in os.walk(imageDir) for file in files]
        fileNumbers = [int((path.split('/')[-1]).split('.')[0]) for path in paths]
        with open(labelDir, 'r') as f:
            labels = [int(label.replace('\n','')) for label in f]
        self.dataPair = dict( [ (fileNumber-1, (paths[i], labels[fileNumber-1])) for i, fileNumber in enumerate(fileNumbers) ] )
        self.dataNum = len(paths)
        self.batchInd = 0
        # image params
        im = np.asarray(Image.open(paths[0]))
        self.inputRow, self.inputCol = inputSize
        self.classNum = classNum
        self.flagColorChannel = False if len(im.shape) == 2 else True
        self.inputChannel = im.shape[2] if self.flagColorChannel else 1
        # pre-processing ## TODO
        self.flagResize = False if inputSize == im.shape[:2] else True ## to be changed
        self.flagNorm = flagNorm ## to be changed
        self.flagWhite = flagWhite ## to be changed
