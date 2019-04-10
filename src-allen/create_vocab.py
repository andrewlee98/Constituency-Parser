
import numpy as np #hello!
import pickle
from utils import *
import math
from datetime import datetime
from matplotlib import pyplot as plt
import os

val_data, test_data, train_data = [], [], []
datapath = 'data/features/'
for file in os.listdir(datapath):
    if file[0] == '.': continue
    elif file[0:2] in {'22'}: val_data.extend(pickle.load(open(datapath + file, 'rb'))) # use folder 22 as validation set
    elif file[0:2] in {'23'}: test_data.extend(pickle.load(open(datapath + file, 'rb'))) # use folder 23 for test
    else: train_data.extend(pickle.load(open(datapath + file, 'rb'))) # training data

vocab = Vocab(train_data + val_data + test_data)
pickle.dump(vocab, open('net_data/vocab.data', 'wb'))

with open('debug/vocab.txt', 'w') as f:
    # f.write("words: " + str(list(map(lambda x: (x, vocab.word2id(x)), vocab.words))) + "\n\n")
    f.write("actions: " + str(vocab.output_act_dict) + "\n\n")
    f.write("labels: " + str(vocab.feat_acts_dict) + "\n\n")
