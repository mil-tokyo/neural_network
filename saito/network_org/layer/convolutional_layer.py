from abstract_layer import AbstractLayer
import neuron
import numpy as np
import activator

class ConvolutionalLayer(AbstractLayer):

    def __init__(self, map_size, map_num, activate_function='sigmoid'):
        self.map_num = map_num
        self.map_size = map_size
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
        next_node = np.dot(self.weight, self.node) + self.bias
        if self.is_softmax:
            next_node = self.ac.activate(next_node,'softmax')
        else:
            next_node = self.ac.activate(next_node)
        return next_node
            

    def back(self, next_derr=None):
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
        self.weight = self.weight - rate * self.dweight
        self.bias = self.bias - rate * self.dbias
