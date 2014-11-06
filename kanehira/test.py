from network import NeuralNetwork
import sys
sys.path.append("../saito/")
import toolbox
import numpy as np

def main():
    ## load dataset and preprocess ##
    from sklearn.datasets import load_digits, fetch_mldata
    from sklearn import preprocessing
    from sklearn.metrics import classification_report
    from sklearn.cross_validation import train_test_split
    dataset = load_digits()
    dataset = fetch_mldata('MNIST original', data_home='.')

    data = preprocessing.normalize(dataset.data, norm='l2')
    lb = preprocessing.LabelBinarizer()
    label = lb.fit_transform(dataset.target)

    train_data, train_label, test_data, test_label = toolbox.load_mnist()
#    train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
#    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

    inp_dim = train_data.shape[1]
    label_dim = label.shape[1]
    print label

    ## network setting ##
    network_setting = read_setting.read_setting()

    ## create Network instance ##
    neuralnet = NeuralNetwork(network_setting)

    ## train Network ##
    neuralnet.train(train_data, train_label)

    ## predict ##
    result = NN.predict(test_data, label_dim)
    print classification_report(test_label, result)


if __name__ == "__main__":
    main()
