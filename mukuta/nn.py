import numpy as np

mu=0
sigma=1
learnparam=0.1

class Layer:
    def update():
        return

class FCLayer(Layer):
    """Full connect layer"""
    def __init__(self,inputnum,outputnum,method):
        self.i=inputnum
        self.o=outputnum
        self.weight=np.random.normal(mu,sigma,(outputnum,inputnum))
        self.diff=grads[method]()

    def forward(self,ivector):
        self.ivector=ivector
        self.ovector=np.dot(self.weight,ivector)
        return self.ovector

    def backward(self,odiff):
        self.diff.append(np.dot(odiff,self.ivector.T))
        return np.dot(self.weight.T,odiff)

    def update(self):
        self.weight=self.diff.update(self.weight)
        
class Gradient:
    """Gradient Descent"""
    def __init__(self):
        self.dnum=0
        self.grad=None

    def append(grad):
        if (self.grad==None):
            self.grad=grad
        else:
            self.grad=self.grad+grad
        self.dnum=self.dnum+1

    def update(weight):
        newweight=weight-learnparam*self.grad/self.dnum
        self.grad=None
        self.dnum=0
        return newweight

class TanhLayer(Layer):
    """apply tanh to linear output"""
    def forward(self,ivector):
        self.ivector=ivector
        self.ovector=np.tanh(self.ivector)
        return self.ovector

    def backward(self,odiff):
        return odiff * (self.ovector+1)/2

class ReLULayer(Layer):
    """apply ReLU to linear output"""
    def forward(self,ivector):
        self.ivector=ivector
        self.ovector=np.max(self.ivector,0)
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

grads={'Grad':Gradient}
layers={'FC':FCLayer, 'Output':OutputLayer, 'Tanh':TanhLayer}

class FCNetwork:
    """Full connected Network"""
    def __init__(self,lst,nonlinear,method):
        self.layers=[]
        for i in range(len(lst)-1):
            self.layers.append(layers['FC'](lst[i],lst[i+1],method))
            self.layers.append(layers[nonlinear]())
        self.layers.append(layers['Output']())

    def regression(self,ivector):
        for layer in range(layers):
            ivector=layer.forward(ivector)
        return ivector

    def backprop(self,ovector):
        for layer in reversed(layers):
            ovector=layer.backward(ovector)

    def update(self):
        for layer in layers:
            layers.update()

    def minibatch(self,feature,label):
        datanum=feature.size[1]
        for i in range(datanum):
            self.regression(feature[:,i])
            self.backprop(label[:,i])
        self.update()


