import numpy as np
#import convolve
from convolve import convolve2d as c2d
from convolve import convolve3d_forward as c3d
from convolve import convolve_recurrent, convolve_recurrent_backward, convolve4d_forward, convolve4d_dweight
from pool import pool2d, pool3d
from scipy.signal import convolve2d
import cProfile
import time
import sys

mode = sys.argv[1]

class convLayer():
    def __init__(self):
        '''
        nothing
        '''
    def convolve_scipy(self,a,b):
        #print 'default convolve'
        ret = np.zeros((b.shape[0],128,128),dtype=np.float64)
        for i in xrange(b.shape[0]):
            for j in xrange(a.shape[0]):
                ret[i] += convolve2d(a[j], b[i,j], 'same')
        #print ret.shape
        return ret

    def convolve_debug(self,a,b):
        #print 'default convolve'
        ret = np.zeros((b.shape[0],128,128),dtype=np.float64)
        for i in xrange(b.shape[0]):
            for j in xrange(a.shape[0]):
                ret[i] += c2d(a[j], b[i,j], mode='same',rotation = 1)
        #print ret.shape
        return ret

    def convolve_self(self,a,b):
        #print 'our convolve'
        ret = c3d(a,b,'same', rotation = 1)
        #print ret.shape
        return ret

if mode == 'recurrent':
    node = np.random.rand(100,13,13,256)
    next_derr = 0.2*np.random.rand(100, 13, 13 ,256)-0.1

    weight = 0.4*np.random.rand(256,256,3,3) - 0.2
    
    # ac = lambda x : 1.0 / (1+np.exp(-x))
    start =time.time()
    node_mine = convolve_recurrent(node,weight)
    end = time.time()
    print end - start
    # print node_mine.shape

    # start = time.time()
    # dweight = np.zeros((256,256,3,3))
    # for i in xrange(dweight.shape[0]):
    #     for j in xrange(dweight.shape[1]):
    #         for k in xrange(node.shape[0] - 1):

    #             tmp = convolve2d(node[k,:,:,j], np.rot90(next_derr[k+1,:,:,i],2), 'same')
    #             dweight[i,j]+=tmp[5:8,5:8]

    # end = time.time()
    # print end - start

    start = time.time()
    dweight_mine = convolve_recurrent_backward(node,next_derr, 1, 1)
    end = time.time()
    print end - start

if mode == 'convolve4d':
    node = np.random.rand(102,13,13,128)
    next_derr = np.random.rand(100,10,10,64)
    dweight = convolve4d_dweight(node,next_derr,mode='valid')
    print np.sum(dweight)

    #weight = 2*np.random.rand(128,256,4,4,3)-1

    # start = time.time()
    # for i in xrange(weight.shape[0]):
    #     for j in xrange(weight.shape[1]):
    #         for depth in xrange(next_node.shape[0]):
    #             for weight_z in xrange(weight.shape[4]):
    #                 next_node[depth,:,:,i] += convolve2d(node[depth+weight_z,:,:,j], np.rot90(weight[i,j,:,:,weight_z],2), mode='valid')
    # end = time.time()
    # print end - start

    # start = time.time()
    # next_node_self = convolve4d_forward(node,weight,mode='valid')
    # end = time.time()
    # print end - start

    # node = np.random.rand(100,5,5,128)
    # weight = 2*np.random.rand(64, 128, 3,3,3) - 1
    # start = time.time()
    # next_node_self = convolve4d_forward(node,weight,mode='valid')
    # end = time.time()
    # print end - start

if mode == 'convolve4d_dweight':
    node = np.random.rand(100,13,13,256)
    next_derr = 2*np.random.rand(98,10,10,128)-1
    start = time.time()
    dweight = convolve4d_dweight(node,next_derr,mode='valid')
    end = time.time()
    print end - start
    print dweight.shape
    #print dweight

if mode == 'pool':
    from pool import pool4d
    a = np.random.rand(4,4,4,2)
    # print a
    # print argmax3d(a, 0,3,0,3,0,3,0)
    # print np.argmax(a[:,:,:,0])

    a = np.random.rand(101,10,10,256)
    print a
    start = time.time()
    max_value, pos = pool4d(a,2,2,2,2,2,2)
    end = time.time()
    print a[:,:,:,1]
    print max_value[:,:,:,1]
    print pos[:,:,:,1]
    # print argmax2d(a[0],86,2,90,2)
    # print a[0,86:88,90:92]
    print end - start
    

if mode == 'conv':
    conv = convLayer()
    a = np.random.rand(10,128,128)
    b = np.random.rand(12,10,5,5)

    #print a
    #print b

    start = time.time()
    for i in range(2):
        ret = conv.convolve_scipy(a,b)
    end = time.time()
    print end - start
    print ret[:,0,0]

    start = time.time()
    for i in range(2):
        ret = conv.convolve_debug(a,b)
    end = time.time()
    print end - start
    print ret[:,0,0]

    start = time.time()
    for i in range(2):
        ret = conv.convolve_self(a,b)
    end = time.time()
    print end - start
    print ret[:,0,0]

    #t.timeit()
    # print 'default convolve'
    # for i in xrange(100):
    #     convolve2d(a,b,'same')


