import random
import numpy as np
from collections import defaultdict
import pickle

class Node:
    def __init__(self, label):
        self.label = label
        self.l = None
        self.r = None

class datum:
    def __init__(self, stack, buff, label):
        self.stack = stack
        self.buff = buff
        self.label = label

# generate sentences from the tree
def inorder_sentence(root, s = ""):
    if not root.l and not root.r:
        s += " " + root.label
        return s
    if root.l:
        s = inorder_sentence(root.l, s)
    if root.r:
        s = inorder_sentence(root.r, s)
    return s

# util to debug
def tree_to_str(root, s = ""):
    sr = ""
    sl = ""
    if root.l:
        sl = tree_to_str(root.l, s)
    if root.r:
        sr = tree_to_str(root.r, s)
    if root and not root.r and not root.l:
        s += " " + clean(root.label)
    elif root.label:
        s += " (" + clean(root.label) + sl + sr + ")"
    return s

def stack_to_str(s):
    if not s: return '[]'
    ret = "["
    for t in s[:-1]:
        ret += tree_to_str(t) + ", "
    ret += tree_to_str(s[-1]) + "]"
    return ret

# level order print for debugging
def level_order(root):
    current_level = [root]
    while current_level:
        print(' '.join(str(node.label) for node in current_level))
        next_level = list()
        for n in current_level:
            if n.l:
                next_level.append(n.l)
            if n.r:
                next_level.append(n.r)
            current_level = next_level

def clean(s): # remove excess space
        s = s.rstrip().lstrip()
        return s

# method for transforming one string into a tree
def parse_tree(tree_str):
    tree_str = clean(tree_str)
    if tree_str[0] == "(": # remove surrounding parentheses
        tree_str = tree_str[1:-1]
    root = Node(tree_str.split()[0]) # set first word as root
    tree_str = tree_str.split()[1:] # remove first (root) word
    tree_str = ' '.join(tree_str) # convert back to string


    stack = [] # use to keep track of parentheses
    nested = False # boolean for if in nested statement
    children = [] # list of strings for children

    i = 0
    while i < len(tree_str): # collect children into list
        # nested parentheses case
        if tree_str[i] == "(":
            stack.append("(")
            if not nested: # save index of outermost left paren
                start_idx = i
            nested = True
        elif tree_str[i] == ")" and nested:
            stack.pop()
            if not stack:
                children.append(tree_str[start_idx:i + 1])
                nested = False

        # handle base case string
        elif tree_str[i] not in "() \n\t" and not nested:
            start_idx = i
            while tree_str[i] not in "() \n\t" and i < len(tree_str) - 1:
                i += 1
            children.append(tree_str[start_idx:i + 1])
        i += 1
    # now the children list is complete, recurse
    children = list(map(clean, children)) # clean whitespace from children
    if len(children) == 1:
        root.l = parse_tree(children[0])
    elif len(children) == 2:
        root.l = parse_tree(children[0])
        root.r = parse_tree(children[1])
    elif len(children) > 2: # binarize case
        root.l = parse_tree(children[0])
        binarize_str = "(" + root.label
        binarize_str += "_inner " if "_inner" not in binarize_str else " "
        binarize_str += ' '.join(children[1:]) + ")"
        root.r = parse_tree(binarize_str) # recursively binarize
    return root


class Vocab:
    def __init__(self, feature_list):
        word_count, actions, labels = defaultdict(int), set(), set()

        for feats in feature_list:
            actions.add(feats[-1])
            for word in feats[12:-1]:
                word_count[word] += 1
            for t in feats[:12]:
                labels.add(t)
        actions = list(actions)
        labels = list(labels)
        words = [word for word in word_count.keys() if word_count[word] > 50]

        self.words = ['<UNK>'] + words
        self.word_dict = {word: i for i, word in enumerate(self.words)}

        self.output_acts = list(actions)
        self.output_act_dict = {a: i for i, a in enumerate(self.output_acts)}

        self.feat_acts = list(labels) + ['<UNK>']
        self.feat_acts_dict = {a: i for i, a in enumerate(self.feat_acts)}

    def tagid2tag_str(self, id):
        return self.output_acts[id]

    def tag2id(self, tag):
        return self.output_act_dict[tag]

    def feat_tag2id(self, tag): # definitely need to figure out a way around UNK
        return self.feat_acts_dict[tag] if tag in self.feat_acts_dict else self.feat_acts_dict['<UNK>']

    def word2id(self, word):
        return self.word_dict[word] if word in self.word_dict else self.word_dict['<UNK>']

    def num_words(self):
        return len(self.words)

    def num_tag_feats(self):
        return len(self.feat_acts)

    def num_tags(self):
        return len(self.output_acts)



#####################################

def rearrange(f):
    # 0(5): label of stack[0] ("<word>" if word)
    # 1(7): leftmost POS
    # 2(9): rightmost POS
    # 3(11): label of stack[1] ("<word>" if word)
    # 4(13): leftmost POS
    # 5(15): rightmost POS
    # 6(17): label of stack[2] ("<word>" if word)
    # 7(19): leftmost POS
    # 8(21): rightmost POS
    # 9(23): label of stack[3] ("<word>" if word)
    # 10(25): leftmost POS
    # 11(27): rightmost POS
    labels = set([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    new_list = []
    for i in range(1, len(f)):
        if i in labels:
            new_list.append(f[i])
    for i in range(1, len(f)):
        if i not in labels: # and i not in prefix:
            new_list.append(f[i])
    new_list.append(f[0]) # append label to end
    return new_list


#########################################

def replace_if_num(s):
    def is_num(s1):
        return s1.replace(',','').replace('.','',1).isdigit()
    return "<num>" if is_num(s) else s

def get_left(t):
    if not t.l.r and not t.l.l: # if l child is unary
        return [t.label, replace_if_num(unindex(t.l.label))]
    else:
        return get_left(t.l)

def get_right(t):
    if not t.l.r and not t.l.l: # if l child is unary
        return [t.label, replace_if_num(unindex(t.l.label))]
    else:
        if t.r: # handle strange case of "(NP (NNP Moscow) ))"
            return get_right(t.r)
        else:
            return get_right(t.l)

def unindex(a):
    return a.split("/")[0].rstrip().lstrip() # assume no words contain "/"

#######################################

def extract_features(d):
    features = []
    stack = d.stack[::-1]
    buff = d.buff

    # top four buffer words
    for i in range(0,4):
        if len(buff) > i:
            features.append(replace_if_num(unindex(buff[i])))
        else:
            features.append("<null>")

        # stack items
    for i in range(0,4):
        if len(stack) > i:
            tree = parse_tree(stack[i])
            if tree.l or tree.r: # label
                features.append(remove_trailing(tree.label))
                features.append("<label>")
            else: # word
                features.append("<word>")
                features.append(replace_if_num(unindex(tree.label)))


            if tree.l and tree.r: # binary rule
                # assume a depth of 3 at least
                features.extend(get_left(tree.l))
                features.extend(get_right(tree.r))
            else:
                features.extend(["<null>"] * 4)
        else:
            features.extend(["<null>"] * 6)

    return features

def remove_trailing(label):
    inner = False
    if label[-1] == label[0] and label[0] == '-': return label[1:-1] #-NONE- case
    if '_inner' in label: inner = True
    s = ((label.split("-")[0]).split('_')[0]).split('=')[0]
    if not s: print(label)
    if inner: s += '_inner'
    return s