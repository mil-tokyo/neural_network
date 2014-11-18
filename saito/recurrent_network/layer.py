import numpy as np

def generate_layer():
    '''
    hello
    '''

class InputLayer():
    def __init__(self, dic_size, hidden_node_size):
        self.node = None
        # input word vector : dic_size * 1
        self.nodes = None
        # input word vector : dic_size * T
        self.weight = 2 * np.random.rand(hidden_node_size, dic_size) - 1
        # U : hidden_node_size * dic_size
        self.dweight = np.zeros(self.weight.shape)

    def reinit(self):
        self.nodes = None

    def forward(self, max_trancate = 1000):
        if self.nodes is None:
            self.nodes = self.node
        else:
            self.nodes = np.c_[self.nodes, self.node]
            if self.nodes.shape[1]>max_trancate:
                self.nodes = self.nodes[:,self.nodes.shape[1]-max_trancate:self.nodes.shape[1]]
        return np.dot(self.weight, self.node)

    def back(self, next_derr = None, max_trancate = 1000):
        if len(self.nodes.shape) == 1:
            trancate = 1
            self.dweight = np.outer(next_derr, self.nodes)
        else:
            trancate = min(max_trancate, self.nodes.shape[1])
            self.dweight = np.dot(next_derr[:,1:trancate], self.nodes[:,1:trancate].T)

    def update(self, alpha, beta):
        self.weight = (1 - beta) * self.weight + alpha * self.dweight
        '''
        update
        '''

class HiddenLayer():
    def __init__(self, dic_size, hidden_node_size, activate_function = 'sigmoid'):
        self.node = None
        self.nodes = None
        self.derr = None
        # nodes : hidden_node_size * T
        # derrs : hidden_node_size * (T - 1)

        self.hidden_weight = 2 * np.random.rand(hidden_node_size, hidden_node_size) - 1
        self.hidden_dweight = np.zeros(self.hidden_weight.shape)
        # W : hidden_node_size * hidden_node_size
        self.out_weight = 2 * np.random.rand(dic_size, hidden_node_size) - 1
        self.out_dweight = np.zeros(self.out_weight.shape)
        # V : dic_size * hidden_node_size

        self.softmax = lambda x : np.exp(x)/sum(np.exp(x))

        if activate_function == 'sigmoid':
            self.ac = lambda x : 1.0 / (1+np.exp(-x))
            self.dac = lambda x : x * (1 - x)
        else:
            error = '''
            it's an unsupported function
            '''
            raise NameError(error)

    def reinit(self):
        self.nodes = None
        self.derr = None
        self.node = None

    def forward(self, max_trancate = 1000):
        if self.nodes is None:
            self.node = self.ac(self.node)
            self.nodes = self.node
        else:
            if len(self.nodes.shape) == 1:
                self.node = self.ac(self.node + np.dot(self.hidden_weight, self.nodes))
            else:
                self.node = self.ac(self.node + np.dot(self.hidden_weight, self.nodes[:,self.nodes.shape[1]-1]))
            self.nodes = np.c_[self.nodes, self.node]
            if self.nodes.shape[1]>max_trancate:
                self.nodes = self.nodes[:,self.nodes.shape[1]-max_trancate:self.nodes.shape[1]]

        return self.softmax(np.dot(self.out_weight, self.node))

    def back(self, next_derr = None, max_trancate = 1000):
        if len(self.nodes.shape) == 1:
            trancate = 1
        else:
            trancate = min(max_trancate, self.nodes.shape[1])
        self.out_dweight = np.outer(next_derr, self.node)
        
        self.derr = np.dot(self.out_weight.T, next_derr) * self.dac(self.node)

        for i in xrange(trancate - 1):
            if i == 0:
                self.derr = np.c_[np.dot(self.hidden_weight.T,self.derr)*self.dac(self.nodes[:,trancate-2-i]), self.derr]
            else:
                self.derr = np.c_[np.dot(self.hidden_weight.T, self.derr[:,0])*self.dac(self.nodes[:,trancate-2-i]), self.derr]

        if trancate != 1:
            self.hidden_dweight = np.dot(self.derr[:,1:trancate], self.nodes[:,0:trancate-1].T)

    def update(self, alpha, beta):
        self.hidden_weight = (1 - beta) * self.hidden_weight + alpha * self.hidden_dweight
        self.out_weight = (1 - beta) * self.out_weight + alpha * self.out_dweight
        '''
        update
        '''



class OutputLayer():
    def __init__(self):
        self.node = None
        self.derr = None
        self.error = None
        # output : dic_size * 1
    def reinit(self):
        self.node = None
        self.derr = None

    def forward(self):
        '''
        nothing to do
        '''

    def back(self, next_derr = None, max_trancate = 1000):
        self.derr = next_derr - self.node
        # next_derr : teacher vector
        self.error = sum(np.power(next_derr - self.node,2))

    def update(self,alpha,beta):
        '''
        nothing to do
        '''
