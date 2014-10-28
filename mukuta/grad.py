import numpy as np



learnparam=0.1
        
class Gradient:
    """Gradient Descent"""
    def __init__(self):
        self.dnum=0
        self.grad=None

    def append(self,grad):
        if (self.grad==None):
            self.grad=grad
        else:
            self.grad=self.grad+grad
        self.dnum=self.dnum+1

    def update(self,weight):
        newweight=weight-learnparam*self.grad/self.dnum
        self.grad=None
        self.dnum=0
        return newweight

dictgrads={'Grad':Gradient}
