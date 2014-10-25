from abstract_layer import AbstractLayer
import neuron
import numpy as np
import activator

class FullyConnectedLayer(AbstractLayer):

    def __init__(self, node_num, activate_function='sigmoid', is_output=False):
        self.node_num = node_num
        self.bias = np.zeros(node_num)
        self.ac = activator.Activator(activate_function)
        self.weight = None
        self.grad = None
        self.dweight = None
        self.dbias = None
        self.is_output = is_output

    def init_weight(self, previous_layer_node):
        self.weight = 2 * np.random.rand(self.node_num, previous_layer_node) - 1
        print "initialize weight"
        print self.weight.shape

    def forward(self, input_node):        
        if self.weight is None:
            self.node = input_node
        else:
            self.node = np.dot(self.weight, input_node) + self.bias
            if self.is_output:
                self.node = self.ac.activate(self.node, 'softmax')
            else:
                self.node = self.ac.activate(self.node)

    def back(self):
        self.derr = self.grad * self.ac.deactivate(self.node)
        self.dbias = self.derr
        return np.dot(self.weight.T, self.derr)

