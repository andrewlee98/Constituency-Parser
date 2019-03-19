import numpy as np
from utils import *
import torch
import pickle
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from datetime import datetime
from matplotlib import pyplot as plt
import os

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



def remove_star(s):
    s = s.split()
    s = list(filter(lambda x: '*' not in x, s))
    return ' '.join(s)

def action(b, s, p):
    error = None

    if p.split()[0] == 'shift':
        if len(p.split()) > 1 and p.split()[1] == 'star':
            s.append(Node('*'))

        # normal shift
        try: s.append(b.pop(0))
        except: error = 'pop on empty buffer'

    elif p.split()[0] == 'unary':
        n = Node(clean(p.split()[1]))
        try:
            n.l = s.pop()
            s.append(n)
        except: error = 'unary on empty stack'

    else: # p.split()[0] == 'binary':
        n = Node(clean(p.split()[1]))
        try:
            n.r, n.l = s.pop(), s.pop()
            s.append(n)
        except: error = 'binary on insufficient stack'

    return b, s, error


if __name__ == '__main__':
    vocab = pickle.load(open('vocab.data', 'rb'))
    # load the network
    net = torch.load('net.pkl')
    net.eval()

    # open treebank for testing
    treepath = "../treebank/treebank_3/parsed/mrg/wsj/23"

    # open file and save as one large string
    text = ""
    for filename in os.listdir(treepath):
        if filename.startswith('.'):
            continue
        with open(treepath + "/" + filename, 'r') as f:
            text += f.read().replace('\n', '')

    tree_string_list = []
    s = []
    start = 0
    for i in range(len(text)):
        if text[i] == "(":
            s.append("(")
        elif text[i] == ")":
            s.pop()
            if not s:
                tree_string_list.append(text[start : i + 1])
                start = i + 1

    # turn tree strings into tree_list
    tree_list = []
    for t in tree_string_list:
        tree_list.append((parse_tree(t[1:-1])))

    # use inorder traveral to generate sentences from trees
    sentences = []
    for t in tree_list:
        sentences.append(remove_star(inorder_sentence(t).lstrip()))

    # testing

    with open('../data/src-allen/tree_pred.txt', 'w') as outfile, open('../data/src-allen/evalb.txt', 'w') as evalb:
        for s, t in zip(sentences, tree_list):
            s = [clean(x) for x in s.split()]
            
            #debug
#             print(' '.join(s) + '\n') # print sentence
            outfile.write(' '.join(s) + '\n\n')

            # construct tree
            buff = list(map(Node, s))
            stack = []
            infinite_loop_count = 0 # terminate after 100 moves
            printed_from_error = False
            while buff or len(stack) > 1: # end when buff consumed & stack has tree
                
                
                # cast to string and predict
                stack, buff = list(map(tree_to_str, stack)), list(map(tree_to_str, buff))
                try: f = extract_features(datum(stack, buff, None))
                except: 
#                     print('feature extraction error')
                    printed_from_error = True
                    break
                    
                print(f)
                word_ids = [vocab.word2id(word_feat) for word_feat in f[12:-1]]
                tag_ids = [vocab.feat_tag2id(tag_feat) for tag_feat in f[0:12]]
                f = word_ids + tag_ids
                pred = vocab.tagid2tag_str(net(torch.LongTensor(rearrange([0] + f)[:-1])))
                # outfile.write(str(f) + ' ' +  pred + '\n')

                # cast back to Node and complete action
                stack, buff = list(map(Node, stack)), list(map(Node, buff))
                buff, stack, error = action(buff, stack, pred)
                if error:
                    # outfile.write(error + '\n')
#                     print('Error: ' + error)
#                     print(stack_to_str(stack) + '\n')
                    outfile.write('Error: ' + error + '\n')
                    outfile.write(stack_to_str(stack) + '\n\n')
                    evalb.write(stack_to_str(stack) + '\n\n')
                    printed_from_error = True
                    break
#                 print(pred + '\n' + stack_to_str(stack) + '\n')
                outfile.write(pred + '\n' + stack_to_str(stack) + '\n\n')
                
                infinite_loop_count += 1
                if infinite_loop_count >= 150: 
#                     print('infinite loop error')
#                     print(stack_to_str(stack) + '\n')
                    outfile.write('infinite loop error' + '\n')
                    outfile.write(stack_to_str(stack) + '\n\n')
                    evalb.write(stack_to_str(stack) + '\n\n')
                    printed_from_error = True
                    break
                
#             if not printed_from_error: print(stack_to_str(stack) + '\n')
            if not printed_from_error: 
                outfile.write(stack_to_str(stack) + '\n\n')
                evalb.write(stack_to_str(stack) + '\n\n')
#             print('GROUND TRUTH:\n' + tree_to_str(t) + '\n')
#             print('-------------------end of sentence-----------------\n')
            outfile.write('GROUND TRUTH:\n' + tree_to_str(t) + '\n\n')
            outfile.write('-------------------end of sentence-----------------\n\n')



