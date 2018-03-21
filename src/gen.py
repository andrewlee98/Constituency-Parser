import os

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


# create stack/buffer actions from sentence and tree
def generate_actions(t, s):

    def match_tree(t1, t2): # subroutine for checking matching trees
        if t1 is None and t2 is None: # base case
            return True
        if t1 and t2 and t1.label == t2.label:
            print("label matched")
        if t1 is not None and t2 is not None:
            return (t1.label == t2.label) and \
            match_tree(t1.l, t2.l) and \
            match_tree(t1.r, t2.r)
        return False

    def label_dfs(root, s0, s1): # subroutine for determining the label
        if match_tree(root.l, s0) and match_tree(root.r, s1): # base case
            print("tree matched")
            return root.label
        if root.l and label_dfs(root.l, s0, s1):
            return label_dfs(root.l, s0, s1)
        if root.r and label_dfs(root.r, s0, s1):
            return label_dfs(root.r, s0, s1)

    actions = [] # returns actions list (seq of stack/buff)
    buff = list(map(Node, s.split()[::-1])) # reverse sentence for O(1) pop
    stack = []

    while buff or len(stack) > 1: # end when buffer consumed & stack has tree
        # try to reduce top two items
        if len(stack) > 1: # reduce
            right = stack.pop() # check this order
            left = stack.pop()
            new_node = Node(label_dfs(t, right, left))
            if new_node != None:
                print("reduce")
                new_node.l = left
                new_node.r = right
                stack.append(new_node)
            else: # shift
                print("shift1")
                stack.append(buff.pop())
        else: # shift
            print("shift")
            stack.append(buff.pop())
        actions.append(stack + ["()"] + buff) # record action

    return actions


if __name__ == '__main__':
    treepath = "../treebank/treebank_3/parsed/mrg/atis/"
    outpath = "../data/"

    # open file and save as one large string
    for filename in os.listdir(treepath):
        with open(treepath + filename, 'r') as f:
            text = f.read().replace('\n', '')

    # split the text into a list of tree strings
    text = text.split("( END_OF_TEXT_UNIT )")
    tree_string_list = []
    for t in text:
        if "@" not in t and len(t) != 0:
            tree_string_list.append(t)

    # turn tree strings into tree_list
    tree_list = []
    for t in tree_string_list:
        tree_list.append(parse_tree(t))

    # use inorder traveral to generate sentences from trees
    sentences = []
    for t in tree_list:
        sentences.append(inorder_sentence(t).lstrip()) # extra space on left

    with open(outpath + 'all.data', 'w') as f:
        for t, s in zip(tree_list, sentences):
            f.write('\n'.join(str(v) for v in generate_actions(t, s)))
            f.write("----------------------")
            break # test 1 tree
