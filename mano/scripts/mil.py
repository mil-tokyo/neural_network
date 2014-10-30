from argparse import ArgumentParser
from datasets import *
from networks import Network


def train(name):
    conf = __import__(name).Setting('train')
    datasetInst = eval(conf.datasetName)(conf.get_dataset_setting())
    networkInst = Network(conf.layers, conf.lossFunc)
    networkInst.train(datasetInst, conf.batchSize, conf.iteration, conf.learningRatio)
    # muriyari
    #conf = __import__(name).Setting('test')
    #datasetInst = eval(conf.datasetName)(conf.get_dataset_setting())
    #print networkInst.test(datasetInst, conf.batchSize)

"""
def test(name):
    config = __import__(name)
    structure = config.Setting()
    structure.
"""


if __name__ == '__main__':

    # python mil.py -m train -n milnet
    parser = ArgumentParser(description='Process Deep Learning.')
    parser.add_argument('-m', type=str, dest='mode', action='store',
                        help='set mode (train or test)', required=True)
    parser.add_argument('-n', type=str, dest='name', action='store',
                        help='set name of your network', required=True)
    args = parser.parse_args()
    eval(args.mode)(args.name)
