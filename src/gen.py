# usage: py gen.py
# creates all.data file which contains sequence of shift reduce actions

import os
from utils import *
import time
import pickle

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
    buff = list(map(Node, s.split()[::-1])) # reverse sentence for O(1) pop
    stack = []

    while buff or len(stack) > 1: # end when buffer consumed & stack has tree
        # print(stack_to_str(stack))

        # write the stack and buffer before action is performed
        st = list(map(lambda x: tree_to_str(x), stack))
        bu = list(map(lambda x: tree_to_str(x), buff[::-1]))
        final_label = ""
        final_action = ""
        # try to reduce top two items
        if len(stack) > 1: # reduce
            left = stack[len(stack) - 2]
            right = stack[len(stack) - 1]
            new_node = Node(binary_label_dfs(t, left, right))
            if new_node.label: # found a matching reduce
                final_label = new_node.label
                final_action = "binary"
                new_node.r = stack.pop()
                new_node.l = stack.pop()
                stack.append(new_node)
            else: # try to unary reduce
                child = stack[len(stack) - 1]
                new_node = Node(unary_label_dfs(t, child))
                if new_node.label: # found a unary reduce
                    final_label = new_node.label
                    final_action = "unary"
                    new_node.l = stack.pop()
                    stack.append(new_node)
                else: # shift
                    final_action = "shift"
                    if "*" in buff[-1].label:
                        final_action += " star"
                    stack.append(buff.pop())
        elif len(stack) == 1: # just try unary reduce
            child = stack[len(stack) - 1]
            new_node = Node(unary_label_dfs(t, child))
            if new_node.label: # found a unary reduce
                final_label = new_node.label
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

        # append all changes
        act = final_action
        lab = final_label
        ret.append(datum(st, bu, act + " " + lab))

    return ret


if __name__ == '__main__':
    t0 = time.time()
    treepath = "../treebank/treebank_3/parsed/mrg/wsj/"
    outpath = "../data/"

    # open file and save as one large string
    text = ""
    for folder in os.listdir(treepath):
        if folder.startswith('.'):
            continue
        for filename in os.listdir(treepath + folder):
            if filename.startswith('.'):
                continue
            with open(treepath + folder + "/" + filename, 'r') as f:
                text += f.read().replace('\n', '')
        #break # test only one folder for speed

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

    tree_list = list(map(lambda x: idx_tree(x)[0], tree_list))

    # use inorder traveral to generate sentences from trees
    sentences = []
    for t in tree_list:
        sentences.append(inorder_sentence(t).lstrip()) # extra space on left

    idx = 0
    output_list = []
    with open(outpath + 'actions.data', 'wb') as f:
        for t, s in zip(tree_list[1:], sentences[1:]):
            dat = generate_actions(t, s)
            output_list.extend(dat)
            if idx % 100 == 0: print(str(idx) + "..." , end = ' ', flush = True)
            idx += 1
            if idx == 10000: break # cut the test data short
        print()
        pickle.dump(output_list, f)

    t1 = time.time()
    total = t1-t0
    print("runtime: " + str(total))
