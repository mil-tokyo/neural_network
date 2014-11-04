import numpy as np
import layer
import convlayer
import network

class ConvNetwork(network.Network):
    """Convolution Network"""
    def __init__(self,lst,nonlinear,method):
        network.Network.__init__(self)
        idim=lst[0][0]
        for l in lst[1]:
            odim=l[0][0]
            patchsize=l[0][1:]
            stepsize=l[1]
            poolsize=l[2]
            self.layers.append(convlayer.ConvLayer(idim,odim,patchsize,stepsize,method))
            self.layers.append(layer.dictlayer[nonlinear]())
            self.layers.append(convlayer.MaxPoolLayer(poolsize))
            idim=odim

        self.layers.append(convlayer.ReshapeLayer())
        tmpvector=network.Network.regression(self,np.zeros(lst[0]))
        idim=tmpvector.shape[0]
        for odim in lst[2]:
            self.layers.append(layer.dictlayer['FC'](idim,odim,method))
            self.layers.append(layer.dictlayer[nonlinear]())
            idim=odim

        self.layers.pop(-1)
        self.layers.append(layer.dictlayer['Output']())
