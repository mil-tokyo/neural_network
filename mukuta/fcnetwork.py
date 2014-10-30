import numpy as np
import layer

class FCNetwork:
    """Full connected Network"""
    def __init__(self,lst,nonlinear,method):
        self.layers=[]
        for i in range(len(lst)-1):
            self.layers.append(layer.dictlayer['FC'](lst[i],lst[i+1],method))
            self.layers.append(layer.dictlayer[nonlinear]())
        self.layers.pop(-1)
        self.layers.append(layer.dictlayer['Output']())

    def regression(self,ivector):
        for layer in self.layers:
            ivector=layer.forward(ivector)
        return ivector

    def backprop(self,ovector):
        for layer in reversed(self.layers):
            ovector=layer.backward(ovector)

    def update(self):
        for layer in self.layers:
            layer.update()

    def minibatch(self,feature,label):
        datanum=feature.shape[0]
        for i in range(datanum):
            self.regression(feature[i])
            self.backprop(label[i])
        self.update()

    def train(self,feature,label,niter,nbatch):
        datanum=feature.shape[0]
        for it in range(niter):
            randinds=np.random.permutation(datanum)
            for i in range(datanum/nbatch):
                inds=randinds[nbatch*i:nbatch*(i+1)]
                self.minibatch(feature[inds],label[inds])




