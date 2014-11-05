# ! c:/Python2.7/python.exe
# -*- coding: utf-8 -*-

""" Network main """

from argparse import ArgumentParser
from datasets import *
from networks import Network

# main function
def run(mode, name):
    conf = __import__(name).Setting(mode)
    datasetInst = eval(conf.datasetName)(conf.get_dataset_setting())
    networkInst = Network(conf.layers, conf.lossFunc)
    getattr(networkInst, mode)(datasetInst, conf.batchSize, conf.iteration, conf.learningRatio, conf.printTiming, conf.weightSavePath)

if __name__ == '__main__':
    # command: python mil.py -m train -n milnet
    parser = ArgumentParser(description='Process Deep Learning.')
    parser.add_argument('-m', type=str, dest='mode', action='store', help='set mode as train or test', required=True)
    parser.add_argument('-n', type=str, dest='name', action='store', help='set name of your network', required=True)
    args = parser.parse_args()
    run(args.mode, args.name)
