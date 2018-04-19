import pickle
from net_properties import *
from utils import *
from network import *

if __name__ == '__main__':
    (vocab, net_properties) = pickle.load(open('../data/vocab_net.data', 'r'))
    network = Network(vocab, net_properties)
    network.load('../networks/net.model')

    writer = open('../data/predictions.data', 'w')
    feature_list = pickle.load(open( "../data/features.data", "rb" ))
    for feature_set in feature_list:
        pred = network.decode(feature_set[:-1])

    writer.close()
