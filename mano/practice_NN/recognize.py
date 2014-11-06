# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Recognize Numbers
Dataset: digits, MNIST """

import numpy as np
from ml_perceptron import MultiLayerPerceptron
from sklearn.datasets import load_digits, fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize, LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report


def classification(database='MNIST'):

    ### TODO ###
    # load dataset
    dataset =
    pixel =
    X =
    labels =
    # Normalize data
    X = X.astype(np.float64)
    X = normalize(X, norm='l2')
    # Split dataset
    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.05)
    # Transform target data format to 1 of K description
    t_train = LabelBinarizer().fit_transform(labels_train)
    ### TODO ###
    # 1, train perceptron


    # 2, recognize test images
    res =

    # Show results
    print confusion_matrix(labels_test, res)
    print classification_report(labels_test, res)

if __name__ == "__main__":
    classification(database='digits')
    classification(database='MNIST')
