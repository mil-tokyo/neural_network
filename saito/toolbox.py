import sys,os,struct
import numpy as np
from array import array

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


    return train_images,train_labels,test_images,test_labels
