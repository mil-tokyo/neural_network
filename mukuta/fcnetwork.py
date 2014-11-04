import numpy as np
import layer
import network

class FCNetwork(network.Network):
    """Full connected Network"""
    def __init__(self,lst,nonlinear,method):
        network.Network.__init(self)
        for i in range(len(lst)-1):
            self.layers.append(layer.dictlayer['FC'](lst[i],lst[i+1],method))
            self.layers.append(layer.dictlayer[nonlinear]())
        self.layers.pop(-1)
        self.layers.append(layer.dictlayer['Output']())

