import dynet as dynet
import random
import matplotlib
matplotlib.use('nbagg')
from matplotlib import pyplot as plt
import numpy as np
import pickle
import time
from utils import *

def train_network():
    t0 = time.time()

    we = 100
    pe = 50
    hidden = 200
    minibatch = 1000
    epochs = 5

    net_properties = NetProperties(we, pe, hidden, minibatch)
    vocab = Vocab("../data/mini/features/train.data")
    
    pickle.dump((vocab, net_properties), open("../data/mini/vocab_net.data", 'wb'))
    network = Network(vocab, net_properties)
    (loss_values, validation_accs, train_accs) = network.train("../data/mini/features/train.data", epochs, "../data/mini/features/validation.data")
    network.save("../data/mini/net.model")

    t1 = time.time()
    total = t1 - t0
    print("runtime: " + str(total))
    return (loss_values, validation_accs, train_accs)

if __name__ == "__main__":
    (loss_values, validation_accs, train_accs) = train_network()

    plt.title("Loss over time")
    plt.xlabel("Minibatch")
    plt.ylabel("Loss")
    plt.plot(*zip(*loss_values))
    plt.savefig('loss.png')


    plt.title("Training/Validation Accuracy")
    plt.xlabel("Minibatch")
    plt.ylabel("Loss")
    x, y = zip(*train_accs)
    plt.plot(x, y, 'g')

    x1, y1 = zip(*validation_accs)
    plt.plot(x1, y1, 'r')
    plt.savefig('accuracy.png')