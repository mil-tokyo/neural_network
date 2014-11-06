from network import NeuralNetwork
import read_setting
import numpy as np

def main():
    import sys
    sys.path.append("../saito/")
    import toolbox
    train_data, train_label, test_data, test_label = toolbox.load_mnist()

    random = np.random.randint(len(train_data), size = 10000)
    train_data = train_data[random, :]
    train_label = train_label[random]
    
    random = np.random.randint(len(test_data), size = 500)
    test_data = test_data[random, :]
    test_label = test_label[random]
    
#    print train_data[1,:].shape
#    train_data = np.hstack((np.ones((train_data.shape[0], 1)), train_data))
#    test_data = np.hstack((np.ones((test_data.shape[0], 1)), test_data))

#    inp_dim = train_data.shape[1]
    label_dim = train_label.shape[1]
#    print label

    ## network setting ##
    network_setting = read_setting.read_setting()

    ## create Network instance ##
    neuralnet = NeuralNetwork(network_setting)

    ## train Network ##
    neuralnet.train(train_data, train_label)

    ## predict ##
    result = neuralnet.predict(test_data, label_dim)
    
    ## evaluate ##
    evaluate(test_label, result)


def evaluate(test_label, result):
    true_num = np.sum(result * test_label)
    test_num = len(test_label)
    accuracy = true_num*1.0 / test_num
    print "Result\n accuracy:{} ({}/{})".format(accuracy, true_num, test_num)


if __name__ == "__main__":
    main()
