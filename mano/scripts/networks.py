# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Network Data structure """

from layers import *

class Network:

    def __init__(self, layers):
        for layerSetting in layers:
            getattr(self, layerSetting.name, layerSetting.layerSetting)



    def forwadComputation(self, 



    def set_layers(self, structure):

        for layer in structure:

        a = eval()(type,elements)
        b = Layer(type,elements)
        a.connect(b)
