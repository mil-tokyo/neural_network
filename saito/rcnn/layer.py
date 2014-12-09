from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.signal import convolve2d
from util.convolve import convolve3d_forward, convolve3d_dweight, convolve3d_derr, convolve_recurrent, convolve_recurrent_backward, convolve4d_forward, convolve4d_dweight, convolve4d_backward
from util.pool import pool4d

def generate_layer():
    '''
    return each layers to network
    '''

class AbstractLayer(object):
    __metaclass__ = ABCMeta

class SpacialPoolingLayer(AbstractLayer):
    '''
    naruhodo
    '''
    def __init__(self, select_frame = 6):
        self.node = None
        self.select_frame = select_frame
        self.next_node_size = None
        self.frame_count = np.zeros(select_frame,dtype=np.int)

    def forward(self):
        self.next_node_size = (self.select_frame, self.node.shape[1], self.node.shape[2], self.node.shape[3])
        self.derr = np.empty(self.node.shape)
        self.next_node = np.empty(self.next_node_size)

        for kernel in xrange(self.node.shape[3]):
            for i in xrange(self.node.shape[1]):
                for j in xrange(self.node.shape[2]):
                    for frame in xrange(self.select_frame):
                        self.frame_count[frame] = self.node.shape[0] * (frame+1) / self.select_frame - self.node.shape[0] * frame / self.select_frame
                        self.next_node[frame,i,j,kernel] = np.sum(self.node[(self.node.shape[0]*frame/self.select_frame) : (self.node.shape[0]*(frame+1)/self.select_frame), i, j, kernel])
                        self.next_node[frame,i,j,kernel] /= float(self.frame_count[frame])

        return self.next_node


    def back(self,next_node = None, next_derr = None):
        next_derr = next_derr.reshape(self.next_node_size)
        tmp_derr = np.empty((next_derr.shape[1], next_derr.shape[2], next_derr.shape[3]))
        frame = 0
        for i in xrange(len(self.frame_count)):
            tmp_derr = next_derr[i] / float(self.frame_count[i])
            for j in xrange(self.frame_count[i]):
                self.derr[frame+j] = tmp_derr
            
            frame += self.frame_count[i]

    def update(self, rates, batch_size = None):
        '''
        nothing to do
        '''
        


class MiddleSelectLayer(AbstractLayer):
    '''
    connect recurrent & convolutinal layer
    '''
    def __init__(self, select_frame = 16):
        self.node = None
        self.weight = None
        self.dweight = None
        self.select_frame = 16
        self.first_col = None

    def forward(self):
        first_col = (self.node.shape[1] - self.select_frame)/2
        next_node = self.node[:,first_col:first_col+self.select_frame]
        self.first_col = first_col
        return next_node

    def back(self,next_node = None, next_derr = None):
        self.derr = np.zeros(self.node.shape)
        self.derr[:,self.first_col:self.first_col+self.select_frame] = next_derr

    def update(self,alpha,beta):
        '''
        nothing to do
        '''

class InputLayer(AbstractLayer):
    def __init__(self, node_size = 4096, hidden_node_size = 256, kernel_shape = (5,5), output_kernel = 128, input_kernel = 256):
        self.node = None # w1~wT  frame * 13 * 13 * input
        self.weight = 2 * np.random.rand(output_kernel, input_kernel, kernel_shape[0], kernel_shape[1]) - 1
        self.dweight = np.zeros(self.weight.shape)
        # U & delta U : output * input * kernel_size(x) * kernel_size(y)

    def forward(self, max_trancate = 1000):
        next_node = np.zeros(self.node.shape[0], self.node.shape[1], self.node.shape[2], self.weight.shape[0])

        ## cython later
        for i in xrange(self.node.shape[0]):
            for j in xrange(self.weight.shape[0]):
                for k in xrange(self.weight.shape[1]):
                    next_node[i,:,:,j] +=  self.myconvolve2d(self.node[i,:,:,k], self.weight[j,k], 'same')
        return next_node
        # ret : U(w1 ~ wT)

    def back(self,next_node = None, next_derr = None):
        self.dweight = np.zeros(self.weight.shape)

        ## cython later
        for j in xrange(self.weight.shape[0]):
            for k in xrange(self.weight.shape[1]):
                for i in xrange(self.node.shape[0]):
                    self.dweight[j,k] += self.myconvolve2d(self.node[i,:,:,k], next_derr[i,:,:j], 'same')

        self.dweight = np.dot(next_derr, self.node.T)

    def update(self, rates, batch_size = None):
        # self.weight = (1 - beta) * self.weight - alpha * self.dweight
        self.weight = self.weight - rates[0] * self.dweight

    def myconvolve2d(self, a, b, mode='full'):
        return convolve2d(a,np.rot90(b,2),mode = mode)

class RecurrentLayer(AbstractLayer):
    def __init__(self, activate_function = 'sigmoid', input_kernel = 256, kernel_shape = (3,3)):
        self.node = None # S : frame * 13 * 13 * input_kernel
        self.derr = None # delta K (S = ac(K))
        self.weight = 0.2 * np.random.rand(input_kernel, input_kernel, kernel_shape[0], kernel_shape[1]) - 0.1
        self.dweight = np.zeros(self.weight.shape)
        self.frame = None
        self.padding_x = (kernel_shape[0]-1)/2
        self.padding_y = (kernel_shape[1]-1)/2
        # W & delta W : output * input * kernel_shape[x] * kernel_shape[y]
        if activate_function == 'sigmoid':
            self.ac = lambda x : 1.0 / (1+np.exp(-x))
            self.dac = lambda x : x * (1 - x)
        else:
            error = '''
            it's an unsupported function
            '''
            raise NameError(error)
    
    def reinit(self):
        self.derr = None
        self.node = None

    def forward(self, max_trancate = 1000):
        self.node = convolve_recurrent(self.node, self.weight)
        # self.node[0] = self.ac(self.node[0])
        # for i in xrange(self.node.shape[0]-1):
        #     tmp_node = np.zeros(self.node.shape[1:4])
        #     for j in xrange(self.weight.shape[0]):
        #         for k in xrange(self.weight.shape[1]):
        #             tmp_node[:,:,j] += self.myconvolve2d(self.node[i,:,:,k], self.weight[j,k], mode='same')
        #     self.node[i+1] = self.ac(self.node[i+1] + tmp_node)
        self.frame = self.node.shape[0]
        return self.node
        # ret : (s1~sT)

    def back(self,next_node = None, next_derr = None):
        self.derr = next_derr * self.dac(self.node)
        # add back propagate from your self
        self.dweight = convolve_recurrent_backward(self.node, self.derr, self.padding_x, self.padding_y)

    def update(self, rates, batch_size = None):
        # self.weight = (1 - beta) * self.weight - alpha * self.dweight
        #self.weight = self.weight - rates[0] / self.node.shape[0] * self.dweight
        self.weight = self.weight - rates[0] * self.dweight

class PoolingLayer(AbstractLayer):
    # kokokara
    def __init__(self, kernel_shape = (2,2,2), stride_shape = (2,2,2), pool_type='MAX'):
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
        next_node, self.pos = pool4d(self.node, self.kernel_shape[0], self.kernel_shape[1], self.kernel_shape[2], self.stride_shape[0], self.stride_shape[1], self.stride_shape[2])
        self.next_node_size = next_node.shape
        return next_node

    def back(self,next_node=None,next_derr=None):
        next_derr = next_derr.reshape(self.next_node_size)
        self.derr = np.zeros(self.node.shape)
        # for i in xrange(next_derr.shape[0]):
        #     self.derr[i,:,:] = self.pos[i,:,:]*np.repeat(np.repeat(next_derr[i,:,:],self.kernel_shape[0],axis=0),self.kernel_shape[1],axis=1)

        self.derr[0:self.pos.shape[0]-self.pos.shape[0]%2] = self.pos[0:self.pos.shape[0]-self.pos.shape[0]%2] * np.repeat(np.repeat(np.repeat(next_derr, self.kernel_shape[0],axis=1),self.kernel_shape[1],axis=2), self.kernel_shape[2], axis=0)

    def update(self, rate, batch_size = 1):
        '''
        nothing to do
        '''

class ConvolutionalLayer(AbstractLayer):

    def __init__(self, num_input, num_output, kernel_shape = (3,3,3), stride = 1, connection = None,mode='valid'):
        self.num_output = num_output
        self.num_input = num_input
        self.kernel_shape = kernel_shape
        # x , y , time
        self.stride = stride
        self.node = None
        self.derr = None
        self.next_node_shape = None
        self.mode = mode
        '''
        initialize weight with random
        '''
        self.weight = 0.2*np.random.rand(num_output, num_input, kernel_shape[0], kernel_shape[1], kernel_shape[2])-0.1
        self.dweight = np.zeros(self.weight.shape)
        if self.stride != 1:
            raise NameError("stride except 1 is unsupported")


    def forward(self):
        # next_node :  time, x, y, output_kernel
        # self.node : time, x, y, input_kernel
        # weight : output_kernel, input_kernel, weight_x, weight_y, weight_z(time)
        next_node = convolve4d_forward(self.node,self.weight,mode=self.mode)
        self.next_node_shape = next_node.shape
        # next_node = np.zeros(self.next_node_shape)
        # for i in xrange(self.weight.shape[0]):
        #     for j in xrange(self.weight.shape[1]):
        #         for depth in xrange(next_node.shape[0]):
        #             for weight_z in xrange(self.weight.shape[4]):
        #                 next_node[depth,:,:,i] += self.myconvolve2d(self.node[depth+weight_z,:,:,j], self.weight[i,j,:,:,weight_z],mode='valid')
        return next_node

    def back(self,next_node, next_derr):
        self.derr = np.zeros(self.node.shape)
        self.dweight = np.zeros(self.dweight.shape)
        next_derr = next_derr.reshape(self.next_node_shape)

        self.derr = convolve4d_backward(next_derr,self.weight,mode='full',rotation=1)

        # for i in xrange(self.weight.shape[0]):
        #     for j in xrange(self.weight.shape[1]):
        #         self.derr[j,:,:] += convolve2d(next_derr[i,:,:],self.weight[i,j,:,:],2), mode='full')

        self.dweight = convolve4d_dweight(self.node, next_derr, mode='valid')
        # for i in xrange(next_derr.shape[0]):
        #     for j in xrange(self.node.shape[0]):
        #         self.dweight[i,j,:,:] += self.myconvolve2d(self.node[j,:,:], next_derr[i,:,:], mode='valid')

    def update(self, rates, batch_size=1):
        #self.weight = self.weight - rates[1] / self.node.shape[0] * self.dweight
        self.weight = self.weight - rates[1] * self.dweight
        #self.dweight.fill(0)

    def myconvolve2d(self, a, b, mode='full'):
        return convolve2d(a,np.rot90(b,2),mode = mode)

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

    def update(self, rates, batch_size = 1):
        self.dweight /= batch_size
        self.dbias /= batch_size
        self.weight = self.weight - rates[2] * self.dweight
        self.bias = self.bias - rates[2] * self.dbias
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

    def update(self, rates, batch_size=1):
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

    def update(self, rates, batch_size=1):
        '''
        nothing
        '''
