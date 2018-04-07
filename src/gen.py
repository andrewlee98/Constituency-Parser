import os
from random import randint

debug_on = False

if debug_on: debug = open("debug.log", "w")

class Node:
    def __init__(self, label):
        self.label = label
        self.l = None
        self.r = None


# method for transforming one string into a tree
def parse_tree(tree_str):
    def clean(s): # remove excess space
        s = s.rstrip().lstrip()
        return s

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

def idx_tree(root, i = 0):
    if not root.l and not root.r: #and "*" not in root.label:
        root.label = root.label + "/" + str(i)
        # if "*" not in root.label:
        #     i += 1
        i += 1
        return (root, i)
    if root.l:
        (root.l, i) = idx_tree(root.l, i)
    if root.r:
        (root.r, i) = idx_tree(root.r, i)
    return (root, i)

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
    ret = "["
    for t in s:
        ret += tree_to_str(t) + " ;;"
    ret += "]"
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
                    stack.append(buff.pop())
                    # try:
                    #     stack.append(buff.pop())
                    # except IndexError:
                    #     print(stack_to_str(stack))
                    #     print(stack_to_str(buff))
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
                stack.append(buff.pop())
                if debug_on: debug.write("stack: " + stack_to_str(stack))
                if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
        else: # shift
            final_action = "shift"
            if debug_on: debug.write("~~~shift3~~~\n\n")
            stack.append(buff.pop())
            if debug_on: debug.write("stack: " + stack_to_str(stack))
            if debug_on: debug.write("\nbuff: " + stack_to_str(buff))
        # append all changes
        stack_seq.append(list(map(tree_to_str, stack)))
        buffer_seq.append(list(map(lambda x: x.label, buff[::-1])))
        final_actions.append(final_action)
        final_labels.append(final_label)

        # print(stack_to_str(stack) + "\n")

    action_str = []
    for s, b, a, l in zip(stack_seq, buffer_seq, final_actions, final_labels):
        out_str = a + " "
        if l:
            out_str += l
        out_str += "\n\nstack:\n" + str(s) + "\n\nbuffer:\n" + str(b) + "\n"
        out_str += "-" * 72 + "\n"
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
            f.write('\n'.join(str(v) for v in generate_actions(t, s)))
            f.write("*" * 96)
            if idx % 100 == 0: print(str(idx) + "..." , end = ' ', flush = True)
            idx += 1
        print()

if debug_on: debug.close()
