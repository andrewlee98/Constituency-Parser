import pickle
from net_properties import *
from utils import *
from network import *

if __name__ == '__main__':
    (vocab, net_properties) = pickle.load(open('../data/vocab_net.data', 'rb'))
    network = Network(vocab, net_properties)
    network.load('../networks/net.model')

    writer = open('../data/predictions.data', 'w')
    feature_list = pickle.load(open( "../data/test.data", "rb" ))

    correct = 0
    shiftstar, tp, fp, fn = 0, 0, 0, 0
    for feature_set in feature_list:
        pred = network.decode(feature_set[:-1])

        # if pred == "shift star":
        #     shiftstar += 1
        #     if feature_set[-1] == "shift star":
        #         tp += 1
        #     else:
        #         fp += 1
        # if feature_set[-1] == "shift star" and pred != "shiftstar":
        #     fn += 1
        #     shiftstar += 1

        if pred == feature_set[-1]:
            correct += 1
    print(float(correct)/len(feature_list))
    # print("tp: " + str(tp))
    # print("fn: " + str(fn))
    # print("fp: " + str(fp))
    # print("total stars: " + str(shiftstar))

    writer.close()
