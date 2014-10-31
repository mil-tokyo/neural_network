from abc import ABCMeta, abstractmethod
import neuron
import numpy as np
import activator


def generate_layer():
    '''
    return each layers to network
    '''

class AbstractLayer(object):
    __metaclass__ = ABCMeta

class FullyConnectedLayer(AbstractLayer):

    def __init__(self, node_num, activate_function='sigmoid'):
        self.node_num = node_num
        self.bias = None
        self.ac = activator.Activator(activate_function)
        self.node = np.zeros(node_num)
        self.weight = None
        self.grad = None
        self.dweight = None
        self.dbias = None

    def init_weight(self, next_layer_node):
        self.weight = 2 * np.random.rand(next_layer_node, self.node_num) - 1
        self.bias = np.zeros(next_layer_node)
        print "initialize weight"
        print self.weight.shape
        

    def forward(self):
        return np.dot(self.weight, self.node) + self.bias
            

    def back(self,next_node = None, next_derr=None):

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
            self.dac = lambda np.sign(x)
        elif activate_function == 'softmax':
            self.ac = lambda x : np.exp(x)/sum(np.exp(x))
            self.dac = lambda x : x
        else:
            error = '''
            it's an unsupported function
            supported function : 
            sigmoid, tanh, arctan, relu, softmax
            '''
            raise NameError(error)

    def forward(self):
        return = self.ac(self.node)
            

    def back(self, next_node, next_derr=None):
        self.derr = 
        if next_derr is None:
            # self.derr = self.grad * self.ac.deactivate(self.node)
            self.derr = self.grad
            return

        self.dbias = next_derr
        self.grad = np.dot(self.weight.T, next_derr)
        # self.dweight = np.array(np.mat(next_derr).T*np.mat(self.node))
        self.dweight = np.outer(next_derr, self.node)
        self.derr = self.grad * self.ac.deactivate(self.node)

    def update(self, rate):
        '''
        nothing
        '''

class OutputLayer(AbstractLayer):

    def __init__(self):
        self.node = None

    def forward(self):
        error = '''
        nothing to forward
        it's an output layer
        '''
        raise NameError(error)
            

    def back(self, label_array):
        self.derr = self.node - label_array

    def update(self, rate):
        '''
        nothing
        '''
