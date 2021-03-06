import numpy as np
import grad
import layer
import convolve3d
import convolveback


mu=0
sigma=0.1

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
        self.ovector=convolve3d.convolve3d(self.ivector,self.weight,self.o,self.osize[0],self.osize[1],self.stepsize[0],self.stepsize[1])
        return self.ovector

    def backward(self,odiff):
        dweight=np.zeros(self.weight.shape)
        idiff=np.zeros(self.ivector.shape)
        dweight,idiff=convolveback.convolveback(odiff,self.weight,self.ivector,self.o,self.osize[0],self.osize[1],self.stepsize[0],self.stepsize[1])
        self.diff.append(dweight)
        return idiff
    def update(self):
        self.weight=self.diff.update(self.weight)

class MaxPoolLayer(layer.Layer):
    """max pool"""
    def __init__(self,patchsize):
        self.patchsize=patchsize

    def forward(self,ivector):
        self.ivector=ivector
        self.osize=[x/z for (x,z) in zip(self.ivector.shape[1:],self.patchsize)]
        lst=[]
        for i in range(2):
            ltmp=[]
            for j in range(self.patchsize[i]):
                ltmp.append(range(j,self.ivector.shape[i+1],self.patchsize[i]))
            lst.append(ltmp)
        tmp=np.amax(np.array([self.ivector[:,l,:] for l in lst[0]]),axis=0)
        self.ovector=np.amax(np.array([tmp[:,:,l] for l in lst[1]]),axis=0)
        self.maxfilter=np.repeat(np.repeat(self.ovector,self.patchsize[0],axis=1),self.patchsize[1],axis=2) <= self.ivector
        return self.ovector

    def backward(self,odiff):
        return np.repeat(np.repeat(odiff,self.patchsize[0],axis=1),self.patchsize[1],axis=2) * self.maxfilter

class ReshapeLayer(layer.Layer):
    """convert 2d array to 1d array"""
    def forward(self,ivector):
        self.shape=ivector.shape
        self.ivector=ivector
        return np.reshape(ivector,np.product(self.shape))

    def backward(self,odiff):
        return np.reshape(odiff,self.shape)




        


