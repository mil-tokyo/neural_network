# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Visualize dataset
Digits, MNIST """

import numpy as np
import pylab as pl
from sklearn.datasets import load_digits, fetch_mldata

def show_data(database='MNIST', numRow=5, numCol=5):
    # Digit data: 1797 samples, 8*8 pixel patch, class 0-9
    # MNIST data: 70000 samples, 28*28 pixel patch, class 0-9
    # x: dataset.data[i], t: dataset.target[i]

    dataset = fetch_mldata('MNIST original', data_home='.') if database=='MNIST' else load_digits()
    ### TO DO ###
    # 1, get patch size i.e. 8 or 28
    pixel = int( np.sqrt(dataset.data.shape[1]) )
    # 2, randomly select image's indice to visualize 
    ind = np.random.random_integers(0, dataset.data.shape[0], numRow*numCol)
    for index, (image, label) in enumerate(np.array(zip(dataset.data, dataset.target))[ind]):
        pl.subplot(numRow, numCol, index + 1)
        pl.axis('off')
        pl.imshow(image.reshape(pixel,pixel), cmap=pl.cm.gray_r)
        pl.title('%i' % label)
    pl.show()

if __name__ == '__main__':
    show_data(database='digits')
    show_data(database='MNIST')
