import pickle
from net_properties import *
from utils import *
from network import *

if __name__ == '__main__':
    we = 100
    pe = 50
    hidden = 200
    minibatch = 1000
    epochs = 5

    net_properties = NetProperties(we, pe, hidden, minibatch)
    vocab = Vocab("../data/train.data")
    pickle.dump((vocab, net_properties), open("../data/vocab_net.data", 'wb'))
    network = Network(vocab, net_properties)
    network.train("../data/train.data", epochs)
    network.save("../networks/net.model")

