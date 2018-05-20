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
    shiftstars, tps, fps, fns, shifts, tpss, fpss, fnss = 0, 0, 0, 0, 0, 0, 0, 0
    for feature_set in feature_list:
        pred = network.decode(feature_set[:-1])
        print("features:" + str(feature_set) + " \npred: " + pred + "\n\n") 

        # count shift stars
        if pred == "shift star ":
            if feature_set[-1] == "shift star ":
                shiftstars += 1
                tpss += 1
            else:
                fpss += 1
        if feature_set[-1] == "shift star " and pred != "shift star ":
            fnss += 1
            shiftstars += 1

        # count shifts
        if pred == "shift ":
            if feature_set[-1] == "shift ":
                shifts += 1
                tps += 1
            else:
                fps += 1
        if feature_set[-1] == "shift " and pred != "shift ":
            fns += 1
            shifts += 1

        # total accuracy
        if pred == feature_set[-1]:
            correct += 1


    print("accuracy: " + str(float(correct)/len(feature_list)))

    print("----------------------------------------")

    precision_ss = tpss / (tpss + fpss)
    recall_ss = tpss / (tpss + fnss)
    print("star precision: " + str(precision_ss))
    print("star recall: " + str(recall_ss))
    print("total stars: " + str(shiftstars))
    print("star F1: " + str( 2*(precision_ss * recall_ss) / (precision_ss + recall_ss) ))

    print("----------------------------------------")

    precision_s = tps / (tps + fps)
    recall_s = tps / (tps + fns)
    print("shift precision: " + str(precision_s))
    print("shift recall: " + str(recall_s))
    print("total shifts: " + str(shifts))
    print("shift F1: " + str( 2*(precision_s * recall_s) / (precision_s + recall_s) ))

    writer.close()
