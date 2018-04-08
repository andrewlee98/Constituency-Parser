# usage: py gen.py
# creates all.data file which contains sequence of shift reduce actions

import os
from utils import *

debug_on = False
if debug_on: debug = open("debug.log", "w")


def idx_tree(root, i = 0, star = 0): # appends indices to tree
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
        s += " " + root.label
    elif root.label:
        s += "(" + root.label + sl + sr + ")"
    return s

def stack_to_str(s):
    ret = ""
    for t in s:
        ret += tree_to_str(t) + list_sep
    ret += ""
    return ret

def buff_to_str(s):
    ret = ""
    for t in s:
        ts = tree_to_str(t)
        if "*" not in ts:
            ret += tree_to_str(t) + list_sep
    ret += ""
    return ret

def remove_star_sentence(s):
    s = s.split()
    s1 = []
    for w in s:
        if "*" not in s:
            s1.append(w)
    return ' '.join(s1)

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
        # debugging
        if debug_on: debug.write("\n$$$$$$$$$start_binary$$$$$$$$$\n")
        if debug_on: debug.write("root: " + tree_to_str(root) + "\n")
        if debug_on: debug.write("s0: " + tree_to_str(s0) + "\n")
        if root.l:
            if debug_on: debug.write("root.l: " + tree_to_str(root.l) + "\n---\n")
        if debug_on: debug.write("s1: " + tree_to_str(s1) + "\n")
        if root.r:
            if debug_on: debug.write("root.r: " + tree_to_str(root.r) + "\n")
        if debug_on: debug.write("\n*********end*********\n")

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
        # debugging
        if debug_on: debug.write("\n$$$$$$$$$start_unary$$$$$$$$$\n")
        if debug_on: debug.write("s0: " + tree_to_str(s0) + "\n")
        if root.l:
            if debug_on: debug.write("child: " + tree_to_str(root.l) + "\n")

        if match_tree(root.l, s0) and root.r == None: # base case
            if debug_on: debug.write("match\n")
            if debug_on: debug.write("*********endsucc*********\n\n")
            return root.label
        if debug_on: debug.write("*********endfail*********\n\n")

        # recursive calls
        child_label = None
        if root.l:
            child_label = unary_label_dfs(root.l, s0)

        if root.r and not child_label:
            child_label = unary_label_dfs(root.r, s0)

        if child_label:
            return child_label

    stack_seq = []
    buffer_seq = []
    final_actions = []
    final_labels = []
    buff = list(map(Node, s.split()[::-1])) # reverse sentence for O(1) pop
    stack = []

    while buff or len(stack) > 1: # end when buffer consumed & stack has tree
        # print(stack_to_str(stack))
        final_label = ""
        final_action = ""
        # try to reduce top two items
        if len(stack) > 1: # reduce
            left = stack[len(stack) - 2] # check this order
            right = stack[len(stack) - 1]
            new_node = Node(binary_label_dfs(t, left, right))
            if new_node.label: # found a matching reduce
                final_label = new_node.label
                final_action = "binary"
                if debug_on: debug.write("~~~binary reduce~~~\n\n")
                new_node.r = stack.pop()
                new_node.l = stack.pop()
                stack.append(new_node)
                if debug_on: debug.write("stack: " + stack_to_str(stack))
                if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
            else: # try to unary reduce
                child = stack[len(stack) - 1]
                new_node = Node(unary_label_dfs(t, child))
                if new_node.label: # found a unary reduce
                    final_label = new_node.label
                    final_action = "unary"
                    if debug_on: debug.write("~~~unary reduce~~~\n\n")
                    new_node.l = stack.pop()
                    stack.append(new_node)
                    if debug_on: debug.write("stack: " + stack_to_str(stack))
                    if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
                else: # shift
                    final_action = "shift"
                    if debug_on: debug.write("~~~shift1~~~\n\n")
                    if "*" in buff[-1].label:
                        final_action += " star"
                    stack.append(buff.pop())
                    if debug_on: debug.write("stack: " + stack_to_str(stack))
                    if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
        elif len(stack) == 1: # just try unary reduce
            child = stack[len(stack) - 1]
            new_node = Node(unary_label_dfs(t, child))
            if new_node.label: # found a unary reduce
                final_label = new_node.label
                final_action = "unary"
                if debug_on: debug.write("~~~unary reduce~~~\n\n")
                new_node.l = stack.pop()
                stack.append(new_node)
                if debug_on: debug.write("stack: " + stack_to_str(stack))
                if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
            else: # shift
                final_action = "shift"
                if debug_on: debug.write("~~~shift2~~~\n\n")
                if "*" in buff[-1].label:
                        final_action += " star"
                stack.append(buff.pop())
                if debug_on: debug.write("stack: " + stack_to_str(stack))
                if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
        else: # shift
            final_action = "shift"
            if debug_on: debug.write("~~~shift3~~~\n\n")
            if "*" in buff[-1].label:
                        final_action += " star"
            stack.append(buff.pop())
            if debug_on: debug.write("stack: " + stack_to_str(stack))
            if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
        # append all changes
        stack_seq.append(stack_to_str(stack))
        buffer_seq.append((buff_to_str(buff[::-1])))
        final_actions.append(final_action)
        final_labels.append(final_label)

        # print(stack_to_str(stack) + "\n")

    action_str = []
    for s, b, a, l in zip(stack_seq, buffer_seq, final_actions, final_labels):
        out_str = a + " "
        if l:
            out_str += l
        out_str += sep + s + sep + b + "\n"
        out_str += action_sep
        action_str.append(out_str)
    return action_str


if __name__ == '__main__':
    treepath = "../treebank/treebank_3/parsed/mrg/wsj/"
    outpath = "../data/"

    # open file and save as one large string
    text = ""
    test_file = os.listdir(treepath)[0]
    print("testing: " + test_file + "/" + os.listdir(treepath + test_file)[0])
    for folder in os.listdir(treepath):
        if folder.startswith('.'):
            continue
        for filename in os.listdir(treepath + folder):
            if filename.startswith('.'):
                continue
            with open(treepath + folder + "/" + filename, 'r') as f:
                text += f.read().replace('\n', '')
        break # test only one folder for speed

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

    tree_list = list(map(lambda x: idx_tree(x)[0], tree_list))

    # use inorder traveral to generate sentences from trees
    sentences = []
    for t in tree_list:
        sentences.append(inorder_sentence(t).lstrip()) # extra space on left

    idx = 0
    with open(outpath + 'all.data', 'w') as f:
        for t, s in zip(tree_list[1:], sentences[1:]):
            #f.write(remove_star_sentence(s) + "\n")
            f.write('\n'.join(str(v) for v in generate_actions(t, s)))
            f.write(tree_sep)
            if idx % 100 == 0: print(str(idx) + "..." , end = ' ', flush = True)
            idx += 1
        print()

if debug_on: debug.close()
