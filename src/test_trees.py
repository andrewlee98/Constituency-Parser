import pickle
from net_properties import *
from utils import *
from network import *
import os

def remove_star(s):
    s = s.split()
    s = list(filter(lambda x: '*' not in x, s))
    return ' '.join(s)

if __name__ == '__main__':
    # load the network
    (vocab, net_properties) = pickle.load(open('../data/vocab_net.data', 'rb'))
    network = Network(vocab, net_properties)
    network.load('../networks/net.model')

    # open treebank for testing
    treepath = "../treebank/treebank_3/parsed/mrg/wsj/24"

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
            s.append("()")
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
        sentences.append(remove_star(inorder_sentence(t).lstrip())) # extra space on left

    print(sentences)

    for s in sentences:
        # construct tree
        buffer = s
        stack = []

    #pred = network.decode(feature_set[:-1])





