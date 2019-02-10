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


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()# Inherited from the parent class nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)  # 1st Full-Connected Layer: 784 (input data) -> 500 (hidden node)
        self.relu = nn.ReLU()  # Non-Linear ReLU Layer: max(0,x)
        self.fc2 = nn.Linear(hidden_size, num_classes) # 2nd Full-Connected Layer: 500 (hidden node) -> 10 (output class)

    def forward(self, x):  # Forward pass: stacking each layer together
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

if __name__ == '__main__':
    input_size = 27        # The image size = 28 x 28 = 784
    hidden_size = 500      # The number of nodes at the hidden layer
    num_classes = 89       # The number of output classes. In this case, from 0 to 9
    num_epochs = 5         # The number of times entire dataset is trained
    batch_size = 100       # The size of input data took for one iteration
    learning_rate = 0.001  # The speed of convergence


    # load data into dataset objects
    train_data = CPDataset('../data/mini/features/train.data')
    test_data = CPDataset('../data/mini/features/validation.data')

    # convert to dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    # instantiate nn
    net = Net(input_size, hidden_size, num_classes)
    # net.cuda()    # You can comment out this line to disable GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # training
    for epoch in range(num_epochs):
        for i, (fvs, labels) in enumerate(train_loader):   # Load a batch of images with its (index, data, class)
            fvs = Variable(fvs)
            labels = Variable(labels)

            optimizer.zero_grad() # Intialize the hidden weight to all zeros
            outputs = net(fvs.float()) # Forward pass: compute the output class given a image
            loss = criterion(outputs, labels) # Compute the loss: difference between the output class and the pre-given label
            loss.backward()   # Backward pass: compute the weight
            optimizer.step()  # Optimizer: update the weights of hidden nodes

            if (i+1) % 100 == 0:  # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                         %(epoch+1, num_epochs, i+1, len(train_data)//batch_size, loss.data))

    # testing
    correct = 0
    total = 0
    for fvs, labels in test_loader:
        fvs = Variable(fvs)
        outputs = net(fvs.float())
        _, predicted = torch.max(outputs.data, 1)  # Choose the best class from the output: The class with the best score
        total += labels.size(0)# Increment the total count
        correct += (predicted == labels).sum() # Increment the correct count

    print('Accuracy: %f %%' % (100 * correct / total))

    # save the net
    # torch.save(net.state_dict(), ‘fnn_model.pkl’)
