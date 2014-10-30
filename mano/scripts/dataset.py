# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Configure file for dataset """

import os
import numpy as np
from PIL import Image
from tensors import Tensor

class MNIST(object):
    """ MNIST dataset class """
    def __init__(self, imageDir, labelDir, inputSize=None):
        # make data pair
        paths = [os.path.join(root,file) for (root,dirs,files) in os.walk(imageDir) for file in files]
        fileNumbers = [int((path.split('/')[-1]).split('.')) for path in paths]
        with open(labelDir, 'r') as f:
            labels = [int(label.replace('\n','')) for label in f]
        self.dataPair = dict( [ (fileNumber, (paths[i], labels[i])) for i, file_number in enumerate(fileNumbers) ] )
        self.dataNum = len(paths)
        # image params
        im = np.asarray(Image.open(paths[0]))
        self.inputRow, self.inputCol = im.shape[0], im.shape[1] if inputSize is None else inpuSize
        self.inputChannel = im.shape[2]
        self.flagResize = False if inputSize is None else True

    def create_batch_input(self, batchSize):
        # randomly choose indice
        imageIndice = np.random.random_integers(0, self.dataNum, batchSize)
        # create input tensor
        inst = Tensor(batchSize, self.inputChanel, self.inputRow, self.inputCol)
        for batchIndex, imageIndex in enumerate(imageIndice):
            im = np.asarray(Image.open(self.dataPair[batchIndex][0]))
            im = resize_image() if flagResize else pass ## TO DO
            new_im = np.zeros(shape=(self.inputChannel, self.inputRow, self.inputCol))
            for color in xrange(self.inputChannel):
                new_im[color,:,:] = im[:,:,color]
            inst.set_image(batchIndex, new_im)
        return inst.get_value()

    def resize_image(self):
        ## TO DO
        pass
