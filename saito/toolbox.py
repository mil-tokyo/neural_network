import sys,os,struct
import numpy as np
from array import array
import math
import cPickle

def load_mnist(data_path = '/data/ishimochi0/dataset/mnist/'):
    fname_train_img = os.path.join(data_path, 'train-images-idx3-ubyte')
    fname_train_lbl = os.path.join(data_path, 'train-labels-idx1-ubyte')
    fname_test_img = os.path.join(data_path, 't10k-images-idx3-ubyte')
    fname_test_lbl = os.path.join(data_path, 't10k-labels-idx1-ubyte')
    
    # load test image
    flbl = open(fname_test_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    test_lbl = array("b", flbl.read())
    flbl.close()
    
    fimg = open(fname_test_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    test_img = array("B", fimg.read())
    fimg.close()
    
    # convert test image
    digits = np.arange(10)
    ind = [ k for k in range(size) if test_lbl[k] in digits]
    N = len(ind)

    test_images = np.zeros((N, rows*cols),dtype=np.uint8)
    test_labels = np.zeros(N,dtype=np.uint8)
    for i in range(len(ind)):
        test_images[i] = np.array(test_img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols])
        test_labels[i] = test_lbl[ind[i]]

    # load training image
    flbl = open(fname_train_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    train_lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_train_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    train_img = array("B", fimg.read())
    fimg.close()

    # convert training image
    digits = np.arange(10)
    ind = [ k for k in range(size) if train_lbl[k] in digits]
    N = len(ind)

    train_images = np.zeros((N, rows*cols),dtype=np.uint8)
    train_labels = np.zeros(N,dtype=np.uint8)
    for i in range(len(ind)):
        train_images[i] = np.array(train_img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ])
        train_labels[i] = train_lbl[ind[i]]

    test_images = test_images / 255.0
    train_images = train_images / 255.0

    # resize images to 2D
    dsize = int(math.sqrt(len(train_images[0])))
    train_images = train_images.reshape(len(train_images),1,dsize,dsize)
    dsize = int(math.sqrt(len(test_images[0])))
    test_images = test_images.reshape(len(test_images),1,dsize,dsize)

    test_labels = np.fromfunction(lambda i,j:j==test_labels[i],(test_labels.size,10),dtype=int)+0
    train_labels = np.fromfunction(lambda i,j:j==train_labels[i],(train_labels.size,10),dtype=int)+0


    return train_images,train_labels,test_images,test_labels


def load_cifar(self,data_path):
    image_list = []
    label_list = []
    for i in xrange(5):
        fo = open(os.path.join(data_path,'data_batch_'+str(i+1)),'rb')
        dict = cPickle.load(fo)
        fo.close()
        image_list.append(dict['data'])
        label_list.append(dict['labels'])
    train_images = np.r_[image_list[0],image_list[1],image_list[2],image_list[3],imag\
e_list[4]]
    train_labels = np.r_[label_list[0],label_list[1],label_list[2],label_list[3],labe\
l_list[4]]
    fo = open(os.path.join(data_path,'test_batch'), 'rb')
    dict= cPickle.load(fo)
    fo.close()
    test_images = dict['data']
    test_labels = dict['labels']

    test_images = test_images / 255.0
    train_images = train_images / 255.0

    # resize images to 2D
    dsize = 32
    train_images = train_images.reshape(len(train_images),3,dsize,dsize)
    test_images = test_images.reshape(len(test_images),3,dsize,dsize)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    test_labels = np.fromfunction(lambda i,j:j==test_labels[i],(len(test_labels)\
,max(test_labels)+1),dtype=int)+0
    train_labels = np.fromfunction(lambda i,j:j==train_labels[i],(len(train_labe\
ls),max(train_labels)+1),dtype=int)
    
    return train_images, train_labels, test_images, test_labels
