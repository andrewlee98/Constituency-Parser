
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
from low_memory_net import *


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
    vocab = pickle.load(open('net_data/vocab.data', 'rb'))
    # load the network
    net = torch.load('net_data/net.pkl')
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

    with open('final_outputs/tree_pred.txt', 'w') as outfile, open('final_outputs/evalb.txt', 'w') as evalb, open('final_outputs/comp_trees.txt','w') as comp_trees:
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

                # print(f)
                f = rearrange([0] + f)[:-1]
                word_ids = [vocab.word2id(word_feat) for word_feat in f[12:]]
                tag_ids = [vocab.feat_tag2id(tag_feat) for tag_feat in f[0:12]]
                f = word_ids + tag_ids
                prediction_vector = net(torch.LongTensor(f).unsqueeze(0))
                # print(prediction_vector)
                # pred_idx = max(enumerate(list(prediction_vector)), key = lambda x: x[1])[0]
                # print(list(prediction_vector))
                _, pred_idx = torch.max(prediction_vector.data, 1)
                # print(pred_idx)

                pred = vocab.tagid2tag_str(pred_idx)
                # print(pred)
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
                comp_trees.write(stack_to_str(stack) + '\n\n')
#             print('GROUND TRUTH:\n' + tree_to_str(t) + '\n')
#             print('-------------------end of sentence-----------------\n')
            outfile.write('GROUND TRUTH:\n' + tree_to_str(t) + '\n\n')
            comp_trees.write('GROUND TRUTH:\n' + tree_to_str(t) + '\n\n')
            outfile.write('-------------------end of sentence-----------------\n\n')
