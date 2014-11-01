from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.signal import convolve2d

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
        '''
        initialize weight with random
        '''        

    def forward(self):
        next_node = self.node
        self.max_pos = self.node
        for i in xrange(len(self.node)):
            next_node[i], self.max_pos[i] = self.pool2d(self.node[i])
        return next_node

    def pool2d(self, node):
        vertical = (node.shape[0] - self.kernel_size)/self.stride+1
        horizontal = (node.shape[1] - self.kernel_size)/self.stride+1
        max_value = np.zeros((vertical, horizontal))
        max_pos = np.zeros((vertical, horizontal))
        for i in xrange(vertical):
            for j in xrange(horizontal):
                max_value[i,j] = np.max(node[i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size])
                max_pos[i,j] = np.argmax(node[i*self.stride:i*self.stride+self.kernel_size,j*self.stride:j*self.stride+self.kernel_size])
                
        return max_value, max_pos

    def back(self,next_node=None, next_derr):
        self.derr = self.node
        for i in xrange(len(next_derr)):
            self.derr[i] = poolback2d(next_derr[i], self.max_pos[i])

    def poolback2d(self, next_derr, max_pos):
        derr = np.zeros(self.node[0].shape)
        for i in xrange(next_derr.shape[0]):
            for j in xrange(next_derr.shape[1]):
                derr[i*self.stride+max_pos[i,j]/self.kernel_size,j*self.stride+max_pos[i,j]/self.kernel_size*self.kernel_size]+=next_derr[i,j]

    def update(self, rate):
        '''
        nothing to do
        '''

class ConvolutionalLayer(AbstractLayer):
    def __init__(self, num_input, num_output, kernel_size = 5, stride = 1):
        self.num_output = num_output
        self.num_input = num_input
        self.kernel_size = kernel_size
        self.stride = stride
        self.node = None
        self.derr = None
        '''
        initialize weight with random
        '''
        self.weight = []
        self.dweight = []
        for i in xrange(num_output):
            self.weight.append(2*np.random.rand(kernel_size,kernel_size)-1)
            self.dweight.append(np.zeros((kernel_size,kernel_size)))

    def forward(self):
        if self.stride != 1:
            raiase NameError("stride except 1 is unsupported")
        next_node = []
        for i in xrange(len(self.weight)):
            next_node.append(convolve2d(self.node[0],self.weight[i],mode='valid'))
            for j in xrange(1,len(self.node)):
                next_node[i] += convolve2d(self.node[j],self.weight[i],mode='valid')
        return next_node
            

    def back(self,next_node, next_derr):

        self.dbias = next_derr
        self.dweight = np.outer(next_derr, self.node)
        self.derr = np.dot(self.weight.T, next_derr)

    def update(self, rate):
        self.weight = self.weight - rate * self.dweight
        self.bias = self.bias - rate * self.dbias

class FullyConnectedLayer(AbstractLayer):
    def __init__(self, node_num, next_node_num):
        self.node_num = node_num
        self.node = np.zeros(node_num)
        self.grad = None
        self.dweight = None
        self.dbias = None

        '''
        initialize weight with random
        '''
        self.weight = 2 * np.random.rand(next_node_num, node_num) - 1
        self.bias = np.zeros(next_node_num)
        print "initialize weight"
        print self.weight.shape
        

    def forward(self):
        return np.dot(self.weight, self.node) + self.bias
            

    def back(self,next_node, next_derr):

        self.dbias = next_derr
        self.dweight = np.outer(next_derr, self.node)
        self.derr = np.dot(self.weight.T, next_derr)

    def update(self, rate):
        self.weight = self.weight - rate * self.dweight
        self.bias = self.bias - rate * self.dbias

class ActivateLayer(AbstractLayer):

    def __init__(self, activate_function='sigmoid'):
        self.node = None

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

    def update(self, rate):
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
        
    def update(self, rate):
        '''
        nothing
        '''
