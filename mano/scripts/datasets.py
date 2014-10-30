# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Configure file for datasets """

import os
import numpy as np
from PIL import Image
from tensors import Tensor

class MNIST(object):
    """ MNIST dataset class """
    def __init__(self, setting):
        imageDir, labelDir, inputSize, classNum, flagNorm, flagWhite = setting
        # make data pair
        paths = [os.path.join(root,file) for (root,dirs,files) in os.walk(imageDir) for file in files]
        fileNumbers = [int((path.split('/')[-1]).split('.')[0]) for path in paths]
        with open(labelDir, 'r') as f:
            labels = [int(label.replace('\n','')) for label in f]
        self.dataPair = dict( [ (fileNumber, (paths[i], labels[i])) for i, fileNumber in enumerate(fileNumbers) ] )
        self.dataNum = len(paths)
        # image params
        im = np.asarray(Image.open(paths[0]))
        self.inputRow, self.inputCol = inputSize
        self.classNum = classNum
        self.flagColorChannel = False if len(im.shape) == 2 else True
        self.inputChannel = im.shape[2] if self.flagColorChannel else 1
        self.flagResize = False if inputSize == im.shape[:2] else True
        self.flagNorm = flagNorm ## to be changed
        # pre-processing ## TODO
        #if self.flagNorm: self.global_contrast_normalization()
        #if self.flagWhite: self.zca_whitening()
        #if self.flagResize: im = self.resize_image(im)

    def create_batch_input(self, batchSize):
        # randomly choose indice
        imageIndice = np.random.random_integers(1, self.dataNum, batchSize)
        # create input tensor
        feat = Tensor(batchSize, self.inputChannel, self.inputRow, self.inputCol)
        target = np.zeros(shape=(batchSize, self.classNum))
        for batchIndex, imageIndex in enumerate(imageIndice):
            im = np.asarray(Image.open(self.dataPair[imageIndex][0]))
            if self.flagNorm: im = self.global_contrast_normalization(im) ## to be changed
            newIm = np.zeros(shape=(self.inputChannel, self.inputRow, self.inputCol))
            for color in xrange(self.inputChannel):
                newIm[color,:,:] = im[:,:,color] if self.flagColorChannel else im[:,:]
            feat.set_image(batchIndex, newIm)
            target[batchIndex,:][self.dataPair[imageIndex][1]] = 1.0 # 1-of-K representation
        return feat.get_value(), target

    ## TO DO
    def resize_image(self, image):
        pass
    """
    def global_contrast_normalization(self, input):
        inputMean = np.apply_along_axis(np.mean, 0, input)
        inputMeanSub = input - inputMean
        print inputMeanSub[:10]
        inputVar = np.apply_along_axis(np.linalg.norm, 0, inputMeanSub)
        print inputVar[:10]
        return inputMeanSub / inputVar
    """
    def global_contrast_normalization(self, input):
        return input / float(np.sum(input * input))

    def zca_whitening(self):
        pass
