from collections import defaultdict
import pickle

class Node:
    def __init__(self, label):
        self.label = label
        self.l = None
        self.r = None

tree_sep = "\n" + "*" * 24 + "\n" # denotes end of one tree's action sequence
action_sep = "\n" + "-" * 24 + "\n" # separates actions from each other
sep = "\n" + "=" * 24 + "\n" # separates action, stack, and buffer in one action
list_sep = ";;" # separates items in stack/buffer

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
    def __init__(self, data_path):
        feature_list = pickle.load(open( "../data/features.data", "rb" ))
        word_count, actions, labels = defaultdict(int), set(), set()

        for feats in feature_list:
            actions.add(feats[-1])
            for word in feats[:12]:
                word_count[word] += 1
            for t in feats[12:-1]:
                labels.add(t)
        words = [word for word in word_count.keys() if word_count[word] > 1]

        self.words = ['<UNK>'] + words
        self.word_dict = {word: i for i, word in enumerate(self.words)}

        self.output_acts = actions
        self.output_act_dict = {a: i for i, a in enumerate(self.output_acts)}
        # for k, v in self.output_act_dict.items():
        #     print(str(k) + " => " + str(v))

        self.feat_acts = labels
        self.feat_acts_dict = {a: i for i, a in enumerate(self.feat_acts)}
        # for k, v in self.feat_acts_dict.items():
        #     print(str(k) + " => " + str(v))

    def tagid2tag_str(self, id):
        return self.output_acts[id]

    def tag2id(self, tag):
        return self.output_act_dict[tag]

    def feat_tag2id(self, tag):
        return self.feat_acts_dict[tag]

    def word2id(self, word):
        return self.word_dict[word] if word in self.word_dict else self.word_dict['<UNK>']

    def num_words(self):
        return len(self.words)

    def num_tag_feats(self):
        return len(self.feat_acts)

    def num_tags(self):
        return len(self.output_acts)








