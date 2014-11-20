from network import NeuralNetwork
import read_setting
import numpy as np
import sys
from sklearn.datasets import fetch_mldata
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

def main():
    sys.path.append("./tools/")
    from tools.load_mnist import load_mnist
    train_data, train_label, test_data, test_label = load_mnist()
    label_dim = train_label.shape[1]

    ## network setting ##
    network_setting = read_setting.read_setting()

    ## create Network instance ##
    neuralnet = NeuralNetwork(network_setting)

    ## train Network ##
    neuralnet.train(train_data, train_label)

    ## predict ##
    result = neuralnet.predict(test_data, label_dim)
    print neuralnet

    ## evaluate ##
    evaluate(test_label, result)

def evaluate(test_label, result):
    true_num = np.sum(result * test_label)
    test_num = len(test_label)
    accuracy = true_num*1.0 / test_num
    print "Result\n accuracy:{} ({}/{})".format(accuracy, true_num, test_num)

if __name__ == "__main__":
    main()
