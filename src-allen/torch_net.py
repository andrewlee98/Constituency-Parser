import numpy as np
import torch
import pickle
from utils import *
import torch.nn.functional as F
import math

class Network:
    def __init__(self):
        # load training, validation data, and vocab
        self.train_data = pickle.load(open('../data/mini/features/train.data', 'rb'))
        self.valid_data = pickle.load(open('../data/mini/features/validation.data', 'rb'))
        self.vocab = Vocab('../data/mini/features/train.data')

        # turn feature vectors from words/labels into numbers
        self.train_data = list(map(self.fv_to_ids, self.train_data))
        self.valid_data = list(map(self.fv_to_ids, self.valid_data))

    def get_data(self):
        # create x_train, y_train, x_valid, y_valid
        x_train = [fv[:-1] for fv in self.train_data]
        y_train = [fv[-1] for fv in self.train_data]
        x_valid = [fv[:-1] for fv in self.valid_data]
        y_valid = [fv[-1] for fv in self.valid_data]

        # cast to torch tensors
        x_train, y_train, x_valid, y_valid = map(
            torch.tensor, (x_train, y_train, x_valid, y_valid))

        return x_train, y_train, x_valid, y_valid

    def fv_to_ids(self, fv):
        word_ids = [self.vocab.word2id(word_feat) for word_feat in fv[12:-1]]
        tag_ids = [self.vocab.feat_tag2id(tag_feat) for tag_feat in fv[0:12]]
        return word_ids + tag_ids

if __name__ == '__main__':
    network = Network()
    x_train, y_train, x_valid, y_valid = network.get_data()

    # output dimensions
    n, c = x_train.shape
    x_train, x_train.shape, y_train.min(), y_train.max()
    print(x_train, y_train)
    print(x_train.shape)
    print(y_train.min(), y_train.max())


    # initialize weights
    weights = torch.randn(27, 89) / math.sqrt(27)
    weights.requires_grad_()
    bias = torch.zeros(89, requires_grad=True)

    def log_softmax(x):
            return x - x.exp().sum(-1).log().unsqueeze(-1)

    def model(xb):
            return log_softmax(xb @ weights + bias)

    # cast to long tensors
    x_train = x_train.long()

    # make batches
    bs = 64  # batch size
    xb = x_train[0:bs]  # a mini-batch from x
    preds = model(xb.float())  # predictions
    preds[0], preds.shape
    print(preds[0], preds.shape)

    def nll(input, target):
            return -input[range(target.shape[0]), target].mean()

    loss_func = nll

    yb = y_train[0:bs]
    print(loss_func(preds, yb))

    def accuracy(out, yb):
        preds = torch.argmax(out, dim=1)
        return (preds == yb).float().mean()

    print(accuracy(preds, yb))

    lr = 0.5  # learning rate
    epochs = 2  # how many epochs to train for

    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            # set_trace()
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb.float())
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                weights -= weights.grad * lr
                bias -= bias.grad * lr
                weights.grad.zero_()
                bias.grad.zero_()
    print(loss_func(model(xb.float()), yb), accuracy(model(xb.float()), yb))


