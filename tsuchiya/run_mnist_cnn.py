#!/usr/bin/python

import sys
import os
import random
import numpy as np
from PIL import Image
from FullyConnectedLayer import FullyConnectedLayer
from OutputLayer import OutputLayer

argv = sys.argv
argc = len(argv)

if (argc <= 2):
    print 'Not enough arguments'
    quit()

mode = argv[1]
if (mode != 'train' and mode != 'test'):
    print 'Invalid mode'
    quit()
    

numbers = range(0,10)
dataset_path = argv[2]
traindata = []
testdata = []
for i in numbers:
    number_path_train = os.path.join(dataset_path, 'train', str(i))
    for filename in os.listdir(number_path_train):
        traindata.append((os.path.join(dataset_path, 'train', str(i), filename), i))

    number_path_test = os.path.join(dataset_path, 'test', str(i))
    for filename in os.listdir(number_path_test):
        testdata.append((os.path.join(dataset_path, 'test', str(i), filename), i))


random.shuffle(traindata)
random.shuffle(testdata)

lr = 0.01
l1 = ConvolutionalLayer((28, 28, 1), lr);
l2 = ConvolutionalLayer((24, 24, 10), lr, stride=1, window=(5, 5))
l2_pool = PoolingLayer((12, 12, 10), lr, overlap=0, window=(2, 2))
l3 = ConvolutionalLayer((10, 10, 12), lr, stride=1, window=(3, 3))
l3_pool = PoolingLayer((5, 5, 12), lr, overlap=0, window=(2, 2))
l4 = FullyConnectedLayer(300, lr)
l5 = FullyConnectedLayer(128, lr)
l6 = OutputLayer(10, lr)

l1.append(l2).append(l2_pool).append(l3).append(l3_pool).append(l4).append(l5).append(l6)

if mode == 'train':

    n_iter = 3
    for j in range(0, n_iter):
        i = 0
        for (image_path, label) in traindata:
            im_pil = Image.open(image_path)
            im = np.asarray(im_pil, dtype=np.float)
            im = im / 255
        
            vec_label = np.zeros(10)
            vec_label[label] = 1

            l1.forward(im.flatten())
            l3.setTrainData(vec_label)
            l3.backward()
    
            i = i + 1

    # test
    n_correct = 0
    for (image_path, label) in testdata:
        im_pil = Image.open(image_path)
        im = np.asarray(im_pil, dtype=np.float)
        im = im/255

        l1.forward(im.flatten())
        ans = np.argmax(l3.getUnits())

        if (label == ans):
            n_correct = n_correct + 1
    
        i = i + 1

    print 1.0*n_correct/i
        

    # test by train set
    
    i = 0
    n_correct = 0
    for (image_path, label) in traindata:
        im_pil = Image.open(image_path)
        im = np.asarray(im_pil, dtype=np.float)
        im = im/255

        l1.forward(im.flatten())
        ans = np.argmax(l3.getUnits())

        if (label == ans):
            n_correct = n_correct + 1
    
        i = i + 1
        """
        if i >= num:
            break
        """

    print 1.0*n_correct/i
    
    
