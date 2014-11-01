import numpy as np
import grad
import layer


mu=0
sigma=1

class ConvLayer(layer.Layer):
    """Convolution Layer"""
    def __init__(self,inputnum,outputnum,patchsize,stepsize,method):
        self.i=inputnum
        self.o=outputnum
        self.patchsize=patchsize
        self.stepsize=stepsize
        self.weight=np.random.normal(mu,sigma,(outputnum,inputnum,patchsize[0],patchsize[1]))
        self.diff=grad.dictgrads[method]()

    def forward(self,ivector):
        self.ivector=ivector
        self.osize=[(x-y)/z+1 for (x,y,z) in zip(ivector.shape[1:],self.patchsize,self.stepsize)]
        self.ovector=np.zeros((self.o,self.osize[0],self.osize[1]))
        for i in range(self.o):
            for j in range(self.osize[0]):
                for k in range(self.osize[1]):
                    self.ovector[i,j,k]=np.sum(self.weight[i] * ivector[:,self.stepsize[0]*j:self.stepsize[0]*j+self.patchsize[0],self.stepsize[1]*k:self.stepsize[1]*k+self.patchsize[1]])
        return self.ovector

    def backward(self,odiff):
        dweight=np.zeros(self.weight.shape)
        idiff=np.zeros(self.ivector.shape)
        for i in range(self.o):
            for j in range(self.osize[0]):
                for k in range(self.osize[1]):
                    dweight[i]+=odiff[i,j,k]*self.ivector[:,self.stepsize[0]*j:self.stepsize[0]*j+self.patchsize[0],self.stepsize[1]*k:self.stepsize[1]*k+self.patchsize[1]]
                    idiff[:,self.stepsize[0]*j:self.stepsize[0]*j+self.patchsize[0],self.stepsize[1]*k:self.stepsize[1]*k+self.patchsize[1]]+=dweight[i]*odiff[i,j,k]
        self.diff.append(dweight)
        return idiff
    def update(self):
        self.weight=self.diff.update(self.weight)



