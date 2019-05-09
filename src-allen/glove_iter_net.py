
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
startTime = datetime.now()

device = torch.device("cuda:0")


# class for loading training/testing datasets
class CPDataset(Dataset):
    def __init__(self, fv_list, vocab):
        # load training, validation data, and vocab
        self.data = np.array(fv_list)
        self.vocab = vocab
        self.length = self.data.shape[0]

        # turn feature vectors from words/labels into numbers
        self.data = list(map(self.fv_to_ids, self.data))

        # create x_train, y_train, x_valid, y_valid
        self.X = [fv[:-1] for fv in self.data]
        self.y = [fv[-1] for fv in self.data]

        # cast to torch tensors
        self.X, self.y= map(torch.tensor, (self.X, self.y))

    def fv_to_ids(self, fv):
        word_ids = [self.vocab.word2id(word_feat) for word_feat in fv[12:-1]]
        tag_ids = [self.vocab.feat_tag2id(tag_feat) for tag_feat in fv[0:12]]
        label_id = self.vocab.tag2id(fv[-1])
        num_fv = word_ids + tag_ids + [label_id]
        # print(num_fv)
        return num_fv

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length

# neural network class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, vocab_size, embedding_dim, vocab, device=None):
        super(Net, self).__init__()# Inherited from the parent class nn.Module

        # handle gpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.word_embeddings = nn.Embedding(len(vocab.word_dict), embedding_dim) # initialize embeddings
        embeds = torch.randn(vocab_size, embedding_dim)
        with open('glove.840B.300d.txt') as f: # replace random init with GloVe
            i = 0
            for line in f:
                fields = line.strip().split(' ')
                if fields[0] in vocab.word_dict: embeds[vocab.word_dict[fields[0]]] = torch.tensor(list(map(float, fields[1:])))
                i += 1
        self.word_embeddings.weight = nn.Parameter(embeds)
        # self.tag_embeddings = nn.Embedding(len(vocab.feat_acts_dict))

        self.fc1 = nn.Linear(input_size * embedding_dim, hidden_size)  # 1st Full-Connected Layer: 2700 (input data) -> 200 (hidden node)
        self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 200 (hidden node) -> 89 (output class)


    def forward(self, x):  # Forward pass: stacking each layer together
        x = x.to(self.device)
        x = x.cuda()
        embeds = self.word_embeddings(x).view(x.shape[0],-1)
        out = self.fc1(embeds)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


def train(net, num_epochs, train_loader, val_loader):
    losses, validations, trains = [], [], []
    for epoch in range(num_epochs):
        for i, (fvs, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
            fvs = Variable(fvs)
            labels = Variable(labels)

            optimizer.zero_grad()               # Initialize the hidden weight to all zeros
            outputs = net(fvs.cuda())          # Forward pass: compute the output class given a image
            loss = criterion(outputs.cuda(), labels.cuda())   # Compute the loss: difference between the output class and the pre-given label
            loss.backward()                     # Backward pass: compute the weight
            optimizer.step()                    # Optimizer: update the weights of hidden nodes

        # Logging
        print('    Epoch [%d/%d], %d Examples, Loss: %.4f'
                 %(epoch+1, num_epochs, len(train_data), loss.data))
        losses.append(loss.data.item()) # plot loss over time

    return net, losses


def test(net, data_loader, vocab, output_file = None):
    total,correct = 0, 0

    write_count = 0
    if output_file: out_stream = open(output_file, 'w')

    for fvs, labels in data_loader:
        fvs = Variable(fvs.cuda())
        outputs = net(fvs.cuda())
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted.cuda() == labels.cuda()).sum()     # Increment the correct counts

        if output_file and write_count < 30:
            labels = list(map(lambda x: x.item(), labels))
            preds = list(map(lambda x: x.item(), predicted))
            labels_word = list(map(lambda x: vocab.tagid2tag_str(x), labels))
            preds_word = list(map(lambda x: vocab.tagid2tag_str(x), preds))

            out_stream.write(str(list(map(str, zip(labels, labels_word, preds, preds_word)))))
        write_count += 1

    return (float(correct) / float(total))


if __name__ == '__main__':
    datapath = 'data/af/'

    # load data into dataset objects (for loading )
    val_data, test_data = [], []
    for file in os.listdir(datapath):
        if file[0:2] in {'22'}: val_data.extend(pickle.load(open(datapath + file, 'rb'))) # use folder 22 as validation set
        elif file[0:2] in {'23'}: test_data.extend(pickle.load(open(datapath + file, 'rb'))) # use folder 23 for test

    # create train/val datasets and vocab
    print('Loading DataSets...')
    vocab = pickle.load(open('net_data/vocab.data', 'rb'))
    val_data = CPDataset(val_data, vocab)
    test_data = CPDataset(test_data, vocab)

    input_size = 28       # 28 features
    hidden_size = 300      # The number of nodes at the hidden layer
    num_classes = len(vocab.output_acts)      # The number of output classes.
    num_epochs = 1
    batch_size = 100
    learning_rate = 0.0001
    vocab_size = 100000    # keep track of some word's embeddings
    embedding_dim = 300   # word embedding size

    # convert to dataloaders
    print('Creating DataLoaders...')
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # instantiate nn
    print('Instantiating NN...')
    net = Net(input_size, hidden_size, num_classes, vocab_size, embedding_dim, vocab) #, torch.device(0))
    net.cuda()    # You can comment out this line to disable GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # run the training method
    trains, validations, losses, folder_loss, folder_acc = [], [] ,[], [], []
    print('Training:')
    for i in range(5):
        for file in sorted(os.listdir(datapath)):
            if file[0:2] not in {'22','23','00','01','24'} and file[0] != '.':
                print('Iteration:' , i+1, 'Training on folder:', file[0:2])
                train_list = pickle.load(open(datapath + file, 'rb'))
                train_data = CPDataset(train_list, vocab)
                train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
                net, new_losses = train(net, num_epochs, train_loader, val_loader)

        # test validation accuracy at the end of every epoch
        val_error = test(net, val_loader, vocab)
        validations.append(val_error)
        train_error = test(net, train_loader, vocab)
        trains.append(train_error)
        print('    Train error: ', train_error, '    Validation error: ', val_error)

        # append training accuracies/losses for graphs
        trains.append(train_error)
        validations.append(val_error)
        losses += new_losses
        folder_loss.append(0 if len(folder_loss) == 0 else 1 + folder_loss[-1])
        folder_acc.append(0 if len(folder_acc) == 0 else 1 + folder_acc[-1])

        print('--------------Iteration Complete-----------------')

    # test
    print('Train Accuracy: %f' % (test(net, train_loader, vocab)))
    print('Test Accuracy: %f' % (test(net, test_loader, vocab, 'debug/preds.txt')))

    # save the net
    torch.save(net, 'net_data/net.pkl')
    print('Training time:', datetime.now() - startTime)

    losses, trains, validations = enumerate(losses), enumerate(trains), enumerate(validations)

    # plot stuff
    plt.title("Loss over time")
    plt.xlabel("Minibatch")
    plt.ylabel("Loss")
    plt.plot(*zip(*losses))
    for xc in folder_loss: plt.axvline(x=xc, color='y', linestyle='--')
    plt.savefig('debug/loss.png')

    plt.figure()
    plt.title("Training/Validation Accuracy")
    plt.xlabel("Minibatch")
    plt.ylabel("Accuracy")
    x1, y1 = zip(*validations)
    plt.plot(x1, y1, 'r')

    x2, y2 = zip(*trains)
    plt.plot(x2, y2, 'g')
    for xc in folder_acc: plt.axvline(x=xc, color='y', linestyle='--')
    plt.savefig('debug/accuracy.png')
