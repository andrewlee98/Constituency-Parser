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
startTime = datetime.now()

device = torch.device("cuda:0")


# class for loading training/testing datasets
class CPDataset(Dataset):
    def __init__(self, datapath):
        # load training, validation data, and vocab
        self.data = np.array(pickle.load(open(datapath, 'rb')))
        self.vocab = Vocab(datapath)
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
        return word_ids + tag_ids

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.length

# neural network class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, vocab_size, embedding_dim, device=None):
        super(Net, self).__init__()# Inherited from the parent class nn.Module

        # handle gpu
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(input_size * embedding_dim, hidden_size)  # 1st Full-Connected Layer: 27 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 89 (output class)

    def forward(self, x):  # Forward pass: stacking each layer together

        x = x.to(self.device)
        x = x.cuda()
        embeds = self.embeddings(x).view(x.shape[0],-1)
        out = self.fc1(embeds)
        out = self.relu(out)
        out = self.fc2(out)
        return out


if __name__ == '__main__':
    input_size = 27        # 27 features
    hidden_size = 500      # The number of nodes at the hidden layer
    num_classes = 89       # The number of output classes.
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001
    vocab_size = 100000    # keep track of some word's embeddings
    embedding_dim = 100   # word embedding size


    # load data into dataset objects
    train_data = CPDataset('../data/mini/features/train.data')
    test_data = CPDataset('../data/mini/features/validation.data')

    # convert to dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # instantiate nn
    net = Net(input_size, hidden_size, num_classes, vocab_size, embedding_dim) #, torch.device(0))
    net.cuda()    # You can comment out this line to disable GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # training
    for epoch in range(num_epochs):
        for i, (fvs, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
            fvs = Variable(fvs)
            labels = Variable(labels)

            optimizer.zero_grad()               # Initialize the hidden weight to all zeros
            outputs = net(fvs.cuda())          # Forward pass: compute the output class given a image
            loss = criterion(outputs.cuda(), labels.cuda())   # Compute the loss: difference between the output class and the pre-given label
            loss.backward()                     # Backward pass: compute the weight
            optimizer.step()                    # Optimizer: update the weights of hidden nodes

            if (i+1) % 100 == 0:  # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                         %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.data))

    # testing
    total,correct = 0, 0
    for fvs, labels in test_loader:
        fvs = Variable(fvs)
        outputs = net(fvs.cuda())
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)                    # Increment the total count
        correct += (predicted.cuda() == labels.cuda()).sum()     # Increment the correct count

    print('Accuracy: %f %%' % (100 * correct / total))

    # save the net
    # torch.save(net.state_dict(), ‘fnn_model.pkl’)
    print(datetime.now() - startTime)
