import numpy as np



learnparam=0.01
        
class Gradient:
    """Gradient Descent"""
    def __init__(self):
        self.dnum=0
        self.grad=0

    def append(self,grad):
        self.grad=self.grad+grad
        self.dnum=self.dnum+1

    def update(self,weight):
        newweight=weight-learnparam*self.grad/self.dnum
        self.grad=0
        self.dnum=0
        return newweight

class AdaGrad(Gradient):
    """Adaptive Gradient"""
    def __init__(self):
        Gradient.__init__(self)
        self.sumgrad=1

    def update(self,weight):
        newweight=weight-learnparam*self.grad/self.dnum/(self.sumgrad ** 0.5)
        self.sumgrad=self.sumgrad + (self.grad * (1.0/self.dnum))**2
        self.grad=0
        self.dnum=0
        return newweight



dictgrads={'Grad':Gradient,'AdaGrad':AdaGrad}
