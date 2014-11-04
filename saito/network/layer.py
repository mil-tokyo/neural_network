from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.signal import convolve2d
from scipy.signal import convolve

def generate_layer():
    '''
    return each layers to network
    '''

class AbstractLayer(object):
    __metaclass__ = ABCMeta

class PoolingLayer(AbstractLayer):
    def __init__(self, kernel_size = 2, stride = 2, pool_type='MAX'):
        self.kernel_size = kernel_size
        self.stride = stride
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
            self.next_node_size = (self.node.shape[0],(self.node.shape[1]-2)/self.stride+1, (self.node.shape[2]-2)/self.stride+1)

        next_node =np.zeros(self.next_node_size)
        self.pos = np.zeros(self.node.shape)

        for i in xrange(len(self.node)):
            next_node[i,:,:], self.pos[i,:,:] = self.pool2d(self.node[i,:,:])
        return next_node

    def pool2d(self, node):
        vertical = (node.shape[0] - self.kernel_size)/self.stride+1
        horizontal = (node.shape[1] - self.kernel_size)/self.stride+1
        max_value = np.zeros((vertical, horizontal))
        pos = np.zeros((node.shape[0],node.shape[1]))

        for i in xrange(vertical):
            for j in xrange(horizontal):
                max_value[i,j] = np.max(node[i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size])
                argmax = np.argmax(node[i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size])
                pos[i*self.stride+argmax/self.kernel_size,j*self.stride+argmax%self.kernel_size] += 1
        return max_value, pos

    def back(self,next_node=None,next_derr=None):
        next_derr = next_derr.reshape(self.next_node_size)
        self.derr = np.zeros(self.node.shape)
        for i in xrange(next_derr.shape[0]):
            self.derr[i,:,:] = self.pos[i,:,:]*np.repeat(np.repeat(next_derr[i,:,:],self.kernel_size,axis=0),self.kernel_size,axis=1)

    def update(self, rate):
        '''
        nothing to do
        '''

class ConvolutionalLayer(AbstractLayer):
    def __init__(self, num_input, num_output, kernel_size = 5, stride = 1, connection = None):
        self.num_output = num_output
        self.num_input = num_input
        self.kernel_size = kernel_size
        self.stride = stride
        self.node = None
        self.derr = None
        self.bias = None
        self.dbias = None
        self.next_node_size = None
        '''
        initialize weight with random
        '''
        self.weight = 2*np.random.rand(num_output, num_input, kernel_size, kernel_size)-1
        self.dweight = np.zeros((num_output, num_input, kernel_size, kernel_size))
        if self.stride != 1:
            raise NameError("stride except 1 is unsupported")


    def forward(self):
        if self.next_node_size is None:
            self.next_node_size = (self.weight.shape[0],(self.node.shape[1]-self.kernel_size)/self.stride+1,(self.node.shape[2]-self.kernel_size)/self.stride+1)
        if self.bias is None:
            self.bias = np.zeros(self.next_node_size)

        next_node = np.zeros(self.next_node_size)
        for i in xrange(self.weight.shape[0]):
            for j in xrange(self.weight.shape[1]):
                next_node[i,:,:] = next_node[i,:,:] + convolve2d(self.node[j,:,:],self.weight[i,j,:,:],mode='valid')
        return next_node

    def back(self,next_node, next_derr):
        self.dbias = next_derr
        self.derr = np.zeros(self.node.shape)
        self.dweight = np.zeros(self.dweight.shape)
        next_derr = next_derr.reshape(self.next_node_size)

        for i in xrange(self.weight.shape[0]):
            for j in xrange(self.weight.shape[1]):
                self.derr[j,:,:] += convolve2d(next_derr[i,:,:],np.rot90(self.weight[i,j,:,:],2),mode='full')
        for i in xrange(next_derr.shape[0]):
            for j in xrange(self.node.shape[0]):
                self.dweight[i,j,:,:] = np.rot90(convolve2d(self.node[j,:,:],np.rot90(next_derr[i,:,:],2),mode='valid'),2)
                # self.dweight[i,j,:,:] = convolve2d(self.node[j,:,:],next_derr[i,:,:],mode='valid')

        
        # self.dweight = np.outer(next_derr, self.node)
        # self.derr = np.dot(self.weight.T, next_derr)
        

    def update(self, rate):
        self.weight = self.weight - rate * self.dweight
        self.bias = self.bias - rate * self.dbias

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
        self.dweight = np.zeros(self.dweight.shape)
        self.dbias = np.zeros(self.dbias.shape)

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

    def update(self, rate, batch_size):
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
        err = np.dot(self.derr,self.derr)/2
        return err
        
    def update(self, rate, batch_size):
        '''
        nothing
        '''
