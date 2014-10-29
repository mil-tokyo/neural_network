from argparse import ArgumentParser
from dataset import MNIST


def train(name):
    config = __import__(name)
    structure = config.Setting()
    networkInst = networks.Network(structure.layers)

    for iter in xrange(structure.get_iteration()):
        input = eval(structure.datasetInst).createBatchInput(structure.batchSize)
        networkInst.forwadComputation(input)
        output = optimization.forward_computation(input)
        input = optimization.baskard_computation(output)


def test(name):
    config = __import__(name)
    structure = config.Setting()
    structure.



if __name__ == '__main__':

    # python mil.py -m train -n milnet.py

    parser = ArgumentParser(description='Process Deep Learning.')
    parser.add_argument('-m', type=str, dest='mode', action='store',
                        help='set mode (train or test)', required=True)
    parser.add_argument('-n', type=str, dest='name', action='store',
                        help='set name of your network', required=True)
    args = parser.parse_args()
    eval(mode)(args.name)
