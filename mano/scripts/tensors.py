# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Tensor Data Strucrue """

import numpy as np

class Tensor(object):
    """ base tensor class """
    def __init__(self, batchSize, channel, row, col):
        self.batch = batchSize
        self.ch = channel
        self.row = row
        self.col = col

        self.value = np.zeros(shape=(batchSize, channel, row, col))

    def set_image(self, batchIndex, image):
        self.value[batchIndex, :, :, :] = iamge

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value
