# creates actions.data file which contains sequence of shift reduce actions
import os
import time
import pickle
from utils import *
import sys

# indexes words in sentence to prevent duplicates
def idx_tree(root, i = 0, star = 0):
    if not root.l and not root.r:
        if "*" not in root.label:
            root.label += "/" + str(i)
            i += 1
        else:
            root.label += "/" + str(star)
            star +=1
        return (root, i, star)
    if root.l:
        (root.l, i, star) = idx_tree(root.l, i, star)
    if root.r:
        (root.r, i, star) = idx_tree(root.r, i, star)
    return (root, i, star)

# create stack/buffer actions from sentence and tree
def generate_actions(t, s):

    def match_tree(t1, t2): # subroutine for checking matching trees
        if t1 is None and t2 is None: # base case
            return True
        if t1 is not None and t2 is not None:
            return (t1.label == t2.label) and \
            match_tree(t1.l, t2.l) and \
            match_tree(t1.r, t2.r)
        return False

    def binary_label_dfs(root, s0, s1): # subroutine for determining the label

        if match_tree(root.l, s0) and match_tree(root.r, s1): # base case
            return root.label
        if root.l:
            left_label = binary_label_dfs(root.l, s0, s1)
            if left_label:
                return left_label
        if root.r:
            right_label = binary_label_dfs(root.r, s0, s1)
            return right_label if right_label else None

    def unary_label_dfs(root, s0): # subroutine for determining the label
        if match_tree(root.l, s0) and root.r == None: # base case
            return root.label

        # recursive calls
        child_label = None
        if root.l:
            child_label = unary_label_dfs(root.l, s0)

        if root.r and not child_label:
            child_label = unary_label_dfs(root.r, s0)

        if child_label:
            return child_label

    ret = []
    buff = s[::-1] #list(map(Node, s.split()[::-1])) # reverse sentence for O(1) pop
    stack = []

    while buff or len(stack) > 1: # end when buffer consumed & stack has tree
        # print(stack_to_str(stack))

        # write the stack and buffer before action is performed
        st = list(stack)
        bu = list(buff[::-1])
        lab = ''
        final_action = ""
        # try to reduce top two items
        if len(stack) > 1: # reduce
            left = stack[len(stack) - 2]
            right = stack[len(stack) - 1]
            new_node = Node(binary_label_dfs(t, left, right))
            if new_node.label: # found a matching reduce
                lab = new_node.label

                final_action = "binary"
                new_node.r = stack.pop()
                new_node.l = stack.pop()
                stack.append(new_node)
            else: # try to unary reduce
                child = stack[len(stack) - 1]
                new_node = Node(unary_label_dfs(t, child))
                if new_node.label: # found a unary reduce
                    lab = new_node.label


                    final_action = "unary"
                    new_node.l = stack.pop()
                    stack.append(new_node)
                else: # shift
                    final_action = "shift"
                    if "*" in buff[-1].label:
                        final_action += " star"
                        # stack.append(Node('*'))
                    stack.append(buff.pop())
        elif len(stack) == 1: # just try unary reduce
            child = stack[len(stack) - 1]
            new_node = Node(unary_label_dfs(t, child))
            if new_node.label: # found a unary reduce
                lab = new_node.label

                final_action = "unary"
                new_node.l = stack.pop()
                stack.append(new_node)
            else: # shift
                final_action = "shift"
                if "*" in buff[-1].label:
                        final_action += " star"
                stack.append(buff.pop())
        else: # shift
            final_action = "shift"
            if "*" in buff[-1].label:
                        final_action += " star"
            stack.append(buff.pop())

        if final_action == 'shift' or final_action == 'shift star':
            d = datum(st, bu, final_action)
            f = rearrange([remove_trailing(d.label)] + extract_features(d))
        else: # unary or binary action
            d = datum(st, bu, final_action + " " + lab)
            f = rearrange([d.label.split()[0] + " " + remove_trailing(d.label.split()[1])] + extract_features(d))

        if f[-1] == 'unary ': print(lab)
        ret.append(f)

    return ret


def treebank_to_actions():
    t0 = time.time()
    treepath = "../treebank/treebank_3/parsed/mrg/wsj/"
    # outpath = "../data/allen/actions/"
    outpath = "data/af/"

    # open file and save as one large string
    for folder in sorted(os.listdir(treepath)):
        if folder.startswith('.') or folder[0:2] in {'00', '01', '24'}: continue

        text = "" # keep one giant text string per folder
        for filename in os.listdir(treepath + folder):
            if filename.startswith('.'): continue
            tree_list = []

            # append all the trees into one string
            with open(treepath + folder + "/" + filename, 'r') as f:
                text += f.read().replace('\n', '')

        # use a stack to parse each tree and append to string list
        s = [] # stack for reading parens
        start = 0
        for i in range(len(text)):
            if text[i].isspace(): continue
            elif text[i] == "(": s.append("(")
            elif text[i] == ")": s.pop()
            if not s:
                tree_list.append(text[start : i + 1])
                start = i + 1

        # convert to tree strings to trees
        tree_list = list(map(lambda x: parse_tree(x[1:-1]), tree_list))

        # index the words in the tree to prevent duplicate word problems
        tree_list = list(map(lambda x: idx_tree(x)[0], tree_list))


        # get sentences
        sentences = [inorder_buffer_gold(x) for x in tree_list]

        output_list = []
        with open(outpath + folder + '_features.data', 'wb') as f:
            for t, s in zip(tree_list, sentences):
                dat = generate_actions(t, s)
                output_list.extend(dat)
            pickle.dump(output_list, f)
        print(folder + "... ")
        # break

    print("runtime: " + str(time.time() - t0))

if __name__ == "__main__":
    treebank_to_actions()

    print('Creating Vocab')

    # vocab creation
    val_data, test_data, train_data = [], [], []
    datapath = 'data/af/'

    for file in sorted(os.listdir(datapath)):
        if file[0] == '.' or file[0:2] in {'00', '01', '24'}: continue
        elif file[0:2] in {'22'}: val_data.extend(pickle.load(open(datapath + file, 'rb'))) # use folder 22 as validation set
        elif file[0:2] in {'23'}: test_data.extend(pickle.load(open(datapath + file, 'rb'))) # use folder 23 for test
        else: train_data.extend(pickle.load(open(datapath + file, 'rb'))) # training data

    vocab = Vocab(train_data + val_data + test_data)
    pickle.dump(vocab, open('net_data/vocab.data', 'wb'))

    with open('debug/vocab.txt', 'w') as f:
        # f.write("words: " + str(list(map(lambda x: (x, vocab.word2id(x)), vocab.words))) + "\n\n")
        f.write("actions: " + str(vocab.output_act_dict) + "\n\n")
        f.write("labels: " + str(vocab.feat_acts_dict) + "\n\n")
