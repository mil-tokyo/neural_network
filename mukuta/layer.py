import numpy as np
import grad



mu=0
sigma=1

class Layer:
    def update(self):
        return

class FCLayer(Layer):
    """Full connect layer"""
    def __init__(self,inputnum,outputnum,method):
        self.i=inputnum
        self.o=outputnum
        self.weight=np.random.normal(mu,sigma,(outputnum,inputnum))
        self.diff=grad.dictgrads[method]()

    def forward(self,ivector):
        self.ivector=ivector
        self.ovector=np.dot(self.weight,ivector)
        return self.ovector

    def backward(self,odiff):
        self.diff.append(np.outer(odiff,self.ivector))
        return np.dot(self.weight.T,odiff)

    def update(self):
        self.weight=self.diff.update(self.weight)

class TanhLayer(Layer):
    """apply tanh to linear output"""
    def forward(self,ivector):
        self.ivector=ivector
        self.ovector=np.tanh(self.ivector)
        return self.ovector

    def backward(self,odiff):
        return odiff * (1-self.ovector**2)/2

class ReLULayer(Layer):
    """apply ReLU to linear output"""
    def forward(self,ivector):
        self.ivector=ivector
        self.ovector=np.maximum(self.ivector,0)
        return self.ovector

    def backward(self,odiff):
        return odiff * np.sign(self.ovector)

class OutputLayer(Layer):
    """output class vector"""
    def forward(self,ivector):
        self.ivector=ivector
        expi=np.exp(self.ivector-np.mean(self.ivector))
        self.ovector=expi/np.sum(expi)
        return self.ovector

    def backward(self,label):
        return self.ovector-label

dictlayer={'FC':FCLayer, 'Output':OutputLayer, 'Tanh':TanhLayer,'ReLU':ReLULayer}
