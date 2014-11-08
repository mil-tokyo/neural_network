import numpy as np
import grad
import layer

mu=0
sigma=0.1

class MemFCLayer(layer.FCLayer):
    """Full connect layer remembering previous input"""
    def __init__(self,inputnum,outputnum,tmem,method):
        layer.FCLayer.__init__(self,inputnum,outputnum,method)
        self.tmem=tmem
        self.memreset()

    def forward(self,ivector):
        self.ivectors=self.ivectors[:-1].append(ivector)
        ovector=np.dot(self.weight,ivector)
        self.ovectors=self.ovectors[:-1].append(ovector)
        return ovector

    def backward(self,odiff,t):
        self.diff.append(np.outer(odiff,self.ivectors[t]))
        return np.dot(self.weight.T,odiff)

    def memreset(self):
        self.ivectors=[]
        self.ovectors=[]
        for i in range(self.tmem):
            self.ivectors.append(np.zeros((self.i,1)))
            self.ovectors.append(np.zeros((self.o,1)))

