import numpy as np #hello!
import pickle
from utils import *
import math
from datetime import datetime
from matplotlib import pyplot as plt
import os

val_data, test_data, train_data = [], [], []
for file in os.listdir('../data/allen/features/'):
    if file[0] == '.': continue
    elif file[0:2] in {'22'}: val_data.extend(pickle.load(open('../data/allen/features/' + file, 'rb'))) # use folder 22 as validation set
    elif file[0:2] in {'23'}: test_data.extend(pickle.load(open('../data/allen/features/' + file, 'rb'))) # use folder 23 for test
    else: train_data.extend(pickle.load(open('../data/allen/features/' + file, 'rb'))) # training data

vocab = Vocab(train_data + val_data + test_data)
pickle.dump(vocab, open('vocab.data', 'wb'))
