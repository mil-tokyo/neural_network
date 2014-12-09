from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.signal import convolve2d
from util.convolve import convolve3d_forward, convolve3d_dweight, convolve3d_derr
from util.pool import pool2d, pool3d

def generate_layer():
    '''
    return each layers to network
    '''

class AbstractLayer(object):
    __metaclass__ = ABCMeta

class PoolingLayer(AbstractLayer):
    def __init__(self, kernel_shape = (2,2), stride_shape = (2,2), pool_type='MAX'):
        self.kernel_shape = kernel_shape
        self.stride_shape = stride_shape
        self.node = None
        self.pool_type = pool_type
        self.derr = None
        self.max_pos = None
        self.next_node_size = None
        '''
        initialize weight with random
        '''        

    def forward(self):
        if self.next_node_size is None:
            self.next_node_size = (self.node.shape[0],(self.node.shape[1]-self.kernel_shape[0])/self.stride_shape[0]+1, (self.node.shape[2]-self.kernel_shape[1])/self.stride_shape[1]+1)
            self.derr = np.empty(self.node.shape)
        next_node, self.pos = pool3d(self.node, self.kernel_shape[0], self.kernel_shape[1], self.stride_shape[0], self.stride_shape[1])
        return next_node

        # for i in xrange(vertical):
        #     for j in xrange(horizontal):
        #         max_value[i,j] = np.max(node[i*self.stride_shape[0]:i*self.stride_shape[0]+self.kernel_shape[0],j*self.stride_shape[1]:j*self.stride_shape[1]+self.kernel_shape[1]])
        #         argmax = np.argmax(node[i*self.stride_shape[0]:i*self.stride_shape[0]+self.kernel_shape[0],j*self.stride_shape[1]:j*self.stride_shape[1]+self.kernel_shape[1]])
        #         pos[i*self.stride_shape[0]+argmax/self.kernel_shape[1],j*self.stride_shape[1]+argmax%self.kernel_shape[1]] += 1
        # return max_value, pos

    def back(self,next_node=None,next_derr=None):
        next_derr = next_derr.reshape(self.next_node_size)
        self.derr.fill(0)
        self.derr = self.pos * np.repeat(np.repeat(next_derr,self.kernel_shape[0],axis = 1),self.kernel_shape[1],axis=2)
        # for i in xrange(next_derr.shape[0]):
        #     self.derr[i,:,:] = self.pos[i,:,:]*np.repeat(np.repeat(next_derr[i,:,:],self.kernel_shape[0],axis=0),self.kernel_shape[1],axis=1)

    def update(self, rate, batch_size = 1):
        '''
        nothing to do
        '''
    def reinit(self):
        '''
        nothing to do
        '''

class ConvolutionalLayer(AbstractLayer):

    def __init__(self, num_input, num_output, kernel_shape = (5,5), stride = 1, connection = None, mode = 'valid'):
        self.num_output = num_output
        self.num_input = num_input
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.node = None
        self.derr = None
        self.next_node_size = None
        self.next_node = None
        self.padding_x = 0
        self.padding_y = 0
        self.padding_flug = 1
        self.mode = mode
        '''
        initialize weight with random
        '''
        if mode == 'same':
            self.padding_x = (kernel_shape[0] - 1)/2
            self.padding_y = (kernel_shape[1] - 1)/2
        elif mode == 'valid':
            self.padding_flug = 0

        self.weight = 2*np.random.rand(num_output, num_input, kernel_shape[0], kernel_shape[1])-1
        self.dweight = np.zeros((num_output, num_input, kernel_shape[0], kernel_shape[1]))
        if self.stride != 1:
            raise NameError("stride except 1 is unsupported")


    def forward(self):
        if self.derr is None:
            self.derr = np.empty(self.node.shape)

        self.next_node = convolve3d_forward(self.node, self.weight, mode = self.mode)
        return self.next_node

    def back(self,next_node, next_derr):
        #self.derr = np.zeros(self.node.shape)
        #self.derr.fill(0)
        #self.dweight.fill(0)
        next_derr = next_derr.reshape(self.next_node.shape)

        self.derr = convolve3d_derr(next_derr, self.weight, mode='full', rotation = 1, padding_x = self.padding_x, padding_y = self.padding_y, padding_flug = self.padding_flug)
        # for i in xrange(self.weight.shape[0]):
        #     for j in xrange(self.weight.shape[1]):
        #         self.derr[j,:,:] += self.myconvolve2d(next_derr[i,:,:],np.rot90(self.weight[i,j,:,:],2),mode='full')

        self.dweight += convolve3d_dweight(self.node, next_derr, mode='valid', padding_x = self.padding_x, padding_y = self.padding_y, padding_flug = self.padding_flug)
        # for i in xrange(next_derr.shape[0]):
        #     for j in xrange(self.node.shape[0]):
        #         self.dweight[i,j,:,:] += self.myconvolve2d(self.node[j,:,:], next_derr[i,:,:], mode='valid')


    def update(self, rate, batch_size=1):
        self.dweight /= batch_size
        self.weight = self.weight - rate * self.dweight
        self.dweight.fill(0)

    def myconvolve2d(self, a, b, mode='full'):
        return convolve2d(a,np.rot90(b,2),mode = mode)

    def reinit(self):
        self.dweight.fill(0)

class FullyConnectedLayer(AbstractLayer):
    def __init__(self, node_num, next_node_num):
        self.node_num = node_num
        self.node = None
        self.grad = None
        self.derr = None

        '''
        initialize weight with random
        '''
        self.weight = 2 * np.random.rand(next_node_num, node_num) - 1
        self.bias = np.zeros(next_node_num)
        self.dweight = np.zeros(self.weight.shape)
        self.dbias = np.zeros(self.bias.shape)
        print "initialize weight"
        print self.weight.shape
        

    def forward(self):
        self.node = self.node.reshape(np.product(self.node.shape))
        return np.dot(self.weight, self.node) + self.bias


    def back(self,next_node, next_derr):
        self.dbias += next_derr
        self.dweight += np.outer(next_derr, self.node)
        self.derr = np.dot(self.weight.T, next_derr)

    def update(self, rate, batch_size = 1):
        self.dweight /= batch_size
        self.dbias /= batch_size
        self.weight = self.weight - rate * self.dweight
        self.bias = self.bias - rate * self.dbias
        self.dweight.fill(0)
        self.dbias.fill(0)

    def reinit(self):
        self.dweight.fill(0)
        self.dbias.fill(0)

class ActivateLayer(AbstractLayer):

    def __init__(self, activate_function='sigmoid'):
        self.node = [None]

        if activate_function == 'sigmoid':
            self.ac = lambda x : 1.0 / (1+np.exp(-x))
            self.dac = lambda x : x * (1 - x)
        elif activate_function == 'tanh':
            self.ac = lambda x : (np.tanh(x)+1)/2.0
            self.dac = lambda x : (1-np.power(x,2))/2.0
        elif activate_function == 'arctan':
            self.ac = lambda x : np.arctan(x) / np.pi + 1/2.0
            self.dac = lambda x : 1/(1+np.power(np.tan(np.pi*(x-1/2.0)),2))/np.pi
        elif activate_function == 'relu':
            self.ac = lambda x : np.maximum(x,0)
            self.dac = lambda x : np.sign(x)
        elif activate_function == 'softmax':
            self.ac = lambda x : np.exp(x)/sum(np.exp(x))
            self.dac = lambda x : 1
        else:
            error = '''
            it's an unsupported function
            supported function : 
            sigmoid, tanh, arctan, relu, softmax
            '''
            raise NameError(error)

    def forward(self):
        return self.ac(self.node)

    def back(self, next_node, next_derr):
        self.derr = next_derr * self.dac(next_node)

    def update(self, rate, batch_size=1):
        '''
        nothing
        '''
    def reinit(self):
        '''
        nothing
        '''


class OutputLayer(AbstractLayer):

    def __init__(self):
        self.node = None
        self.derr = None

    def forward(self):
        error = '''
        nothing to forward
        it's an output layer
        '''
        raise NameError(error)

    def back(self, label_array):
        self.derr = self.node - label_array
        err = sum(-label_array * np.log(self.node))
        #err = np.dot(self.derr,self.derr)/2
        return err

    def update(self, rate, batch_size=1):
        '''
        nothing
        '''

    def reinit(self):
        '''
        nothing
        '''
