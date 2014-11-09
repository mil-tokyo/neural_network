import numpy as np
import layer
import memlayer

class RecurrentNetwork:
    """Recurrent Network"""
    def __init__(self,inputdim,latentdim,tmem,nonlinear,method):
        self.wordmaplayer=memlayer.MemFCLayer(inputdim,latentdim,tmem,method)
        self.latentmaplayer=memlayer.MemFCLayer(latentdim,latentdim,tmem,method)
        self.outmaplayer=layer.FCLayer(latentdim,inputdim,method)
        self.nonlinearlayer=layer.dictlayer[nonlinear]()
        self.outputlayer=layer.dictlayer['Output']()
        self.tmem=tmem

    def train(self,ivectors):
        self.wordmaplayer.memreset()
        self.latentmaplayer.memreset()
        wordnum=ivectors.shape[0]
        word=ivectors[0]
        ivec =self.wordmaplayer.forward(word)
        for t in range(1,wordnum):
            s = self.nonlinearlayer.forward(ivec)
            out=self.outmaplayer.forward(s)
            y = self.outputlayer.forward(out)
            word = ivectors[t]
            self.backward(word)
            self.update()
            ivec = self.wordmaplayer.forward(word) + self.latentmaplayer.forward(s)

    def backward(self,word):
        odiff=self.outputlayer.backward(word)
        odiff=self.outmaplayer.backward(odiff)
        for i in range(self.tmem):
            odiff = self.nonlinearlayer.backward(odiff)
            self.wordmaplayer.backward(odiff,i)
            odiff = self.latentmaplayer.backward(odiff,i)

    def update(self):
        self.wordmaplayer.update()
        self.latentmaplayer.update()
        self.outmaplayer.update()

