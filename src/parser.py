import pickle
from net_properties import *
from utils import *

if __name__ == '__main__':
    we = 100
    pe = 50
    hidden = 200
    minibatch = 1000
    net_properties = NetProperties(we, pe, hidden, minibatch)

    vocab = Vocab("../data/features.data")
