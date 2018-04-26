import pickle
from net_properties import *
from utils import *
from network import *
import time



if __name__ == '__main__':
    t0 = time.time()

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

    t1 = time.time()
    total = t1 - t0
    print("runtime: " + str(total))

