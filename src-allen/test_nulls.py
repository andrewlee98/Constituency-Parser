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
from glove_iter_net import *



class n_Node:
    def __init__(self, label):
        self.label = label
        self.children = []

def debinarize_tree(b_t):
    n_t = n_Node(b_t.label.split('_')[0]) # remove _inner
    if b_t.l: n_t.children.append(b_t.l)

    if b_t.r:
        right = b_t.r
        while right.label[-5:] == 'inner' and right.label.split('_')[0] == n_t.label.split('_')[0]:
            n_t.children.append(right.l)
            right = right.r
        n_t.children.append(right)

    n_t.children = [debinarize_tree(child) for child in n_t.children]

    return n_t

def n_tree_to_str(root, s = ""):
    # base case
    if not root.children: s += " " + clean(root.label)
    elif root.label:
        s += " (" + clean(root.label) + ' '.join([n_tree_to_str(c) for c in root.children]) + ")"
    return s



def remove_sentence_star(s):
    s = s.split()
    s = list(filter(lambda x: '*' not in x, s))
    return ' '.join(s)


def action(b, s, p):
    error = None

    if p.split()[0] == 'shift':
        if s and (not s[-1].l) and (not s[-1].r): return b, s, 'shift on word at end of stack'
        if len(p.split()) > 1:
            print('s', end = '')
            n = Node('-NONE-')
            n.l = Node(p.split()[1] + '^*')
            s.append(n)

        # normal shift
        try: s.append(b.pop(0))
        except: error = 'pop on empty buffer'

    elif p.split()[0] == 'unary':
        if len(s) >= 1:
            n = Node(clean(p.split()[1]))
            n.l = s.pop()
            s.append(n)
        else: error = 'unary on empty stack'

    else: # p.split()[0] == 'binary':
        if s and (not s[-1].l) and (not s[-1].r): return b, s, 'binary on word at end of stack'

        if len(s) >= 2:
            n = Node(clean(p.split()[1]))
            n.r, n.l = s.pop(), s.pop()
            s.append(n)
        else: error = 'binary on insufficient stack'

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
        if text[i] == "(": s.append("(")
        elif text[i] == ")":
            s.pop()
            if not s:
                tree_string_list.append(text[start : i + 1])
                start = i + 1

    # turn tree strings into tree_list
    tree_list = []
    for t in tree_string_list: tree_list.append((parse_tree(t[1:-1])))

    # use inorder traveral to generate sentences from trees
    sentences = []
    for t in tree_list: sentences.append(inorder_buffer_gold_no_null(t))

    # testing
    with open('final_outputs/comp_trees_with_nulls.txt','w') as comp_trees: #open('EVALB/my.tst','w') as tst, open('EVALB/my.gld','w') as gld:
        count = 0
        for s, t, tree_string in zip(sentences, tree_list, tree_string_list):
            if count % 100 == 0: print(count)
            count += 1
            # if count != 328: continue

            # s = [clean(x) for x in s.split()]
            max_depth_stack = [] # keeps track of how many consecutive unary's were done

            # # construct tree
            # buff = list(map(Node, s))
            buff = s
            stack = []
            infinite_loop_count = 0 # terminate after 100 moves

            while buff or len(stack) > 1: # end when buff consumed & stack has tree

                try: f = extract_features(datum(stack, buff, None))
                except Exception as e: print('feature extraction error: ', e, stack_to_str(stack))


                f = rearrange([0] + f)[:-1]
                word_ids = [vocab.word2id(word_feat) for word_feat in f[12:]]
                tag_ids = [vocab.feat_tag2id(tag_feat) for tag_feat in f[0:12]]
                f = word_ids + tag_ids
                prediction_vector = net(torch.LongTensor(f).unsqueeze(0))

                for pred_idx in torch.topk(prediction_vector.data, 10)[1][0]:
                    pred = vocab.tagid2tag_str(pred_idx)
                    if len(max_depth_stack) > 2 and max_depth_stack[-1].split()[0] == max_depth_stack[-2].split()[0] == pred.split()[0] == 'unary': continue
                    buff, stack, error = action(buff, stack, pred) # leaves buff and stack unchanged if error
                    if not error: break
                if error: print('Cycled through and still has errors')
                max_depth_stack.append(pred)
                # if count == 328: print(stack_to_str(buff), stack_to_str(stack), pred, '\n\n')

                infinite_loop_count += 1
                if infinite_loop_count >= 500:
                    print('infinite?')
                    print(stack_to_str(stack))
                    break

            comp_trees.write(#'Prediction:\n' + stack_to_str(stack)[1:-1] + '\n\n' +
                # 'GROUND TRUTH:\n' + tree_to_str(t) + '\n\n' +
                'Debinarized:\n(' +  n_tree_to_str(debinarize_tree(stack[0])) + ')\n\n' +
                'Plaintext Ground Truth:\n' + ' '.join(tree_string.split()) + '\n\n' +
                '-------------------end of sentence-----------------\n\n')

            # tst.write('(' +  n_tree_to_str(debinarize_tree(stack[0])) + ')\n')
            # gld.write(' '.join(tree_string.split()) + '\n')


