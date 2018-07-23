import pickle
from net_properties import *
from utils import *
from network import *
import os
from features import *

def remove_star(s):
    s = s.split()
    s = list(filter(lambda x: '*' not in x, s))
    return ' '.join(s)

def action(b, s, p):
    error = None

    if p.split()[0] == 'shift':
        if len(p.split()) > 1 and p.split()[1] == 'star':
            s.append(Node('*'))
        elif b:
            s.append(b.pop(0))
        else:
            error = 'pop on empty buffer'

    elif p.split()[0] == 'unary':
        n = Node(clean(p.split()[1]))
        if s:
            n.l = s.pop()
            s.append(n)
        else:
            error = 'unary on empty stack'

    else: # p.split()[0] == 'binary':
        n = Node(clean(p.split()[1]))
        if s:
            n.l, n.r = s.pop(), s.pop()
            s.append(n)
        else: error = 'binary on insufficient stack'

    return b, s, error


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
    with open('../data/tree_pred.txt', 'w') as outfile:
        for s in sentences:
            s = [clean(x) for x in s.split()]
            outfile.write(str(s) + '\n')

            # construct tree
            buff = list(map(Node, s))
            stack = []
            while buff or len(stack) > 1: # end when buff consumed & stack has tree

                # cast to string and predict
                stack, buff = list(map(tree_to_str, stack)), list(map(tree_to_str, buff))
                f = extract_features(datum(stack, buff, None))
                pred = network.decode(rearrange([0] + f)[:-1])
                outfile.write(str(f) + ' ' +  pred + '\n')

                # cast back to Node and complete action
                stack, buff = list(map(Node, stack)), list(map(Node, buff))
                buff, stack, error = action(buff, stack, pred)
                if error:
                    outfile.write(error + '\n')
                    break

            outfile.write(tree_to_str(stack[-1]) +  '\n\n')
