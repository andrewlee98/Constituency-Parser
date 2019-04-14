import numpy as np
import torch
import pickle
from utils import *
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from datetime import datetime
from matplotlib import pyplot as plt
import os
from low_memory_net import *
startTime = datetime.now()

device = torch.device("cuda:0")


def move_test(net, data_loader, vocab, output_file = None):
    total, correct = 0, 0

    write_count = 0
    if output_file: out_stream = open(output_file, 'w')

    inner_list = []

    for fvs, labels in data_loader:
        fvs = Variable(fvs.cuda())
        outputs = net(fvs.cuda())
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score


        pred_strings = list(map(lambda x: vocab.tagid2tag_str(x), predicted))
        labels_strings = list(map(lambda x: vocab.tagid2tag_str(x), labels))
        pred_list = zip(pred_strings, labels_strings)
        inner_preds = list(filter(lambda x: x[1] == 'unary DT', pred_list))
        # if inner_preds: print(inner_preds)
        inner_list.extend(inner_preds)


        total += labels.size(0)                    # Increment the total count
        correct += (predicted.cuda() == labels.cuda()).sum()     # Increment the correct counts

    with open(output_file, 'w') as f: f.write(str(inner_list))

    return (float(correct) / float(total))


if __name__ == '__main__':
    batch_size = 100

    test_data = []
    for file in os.listdir('data/features/'):
        if file[0:2] in {'23'}: test_data.extend(pickle.load(open('data/features/' + file, 'rb'))) # use folder 23 for test


    vocab = pickle.load(open('net_data/vocab.data', 'rb'))
    test_data = CPDataset(test_data, vocab)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    net = torch.load('net_data/net.pkl')
    net.eval()

    print('Test Accuracy: %f' % (move_test(net, test_loader, vocab, 'debug/inner_debug.txt')))