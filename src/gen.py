import os

debug = open("debug.log", "w")

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

# util to debug
def print_tree(root, s = ""):
    sr = ""
    sl = ""
    if root.l:
        sl = print_tree(root.l, s)
    if root.r:
        sr = print_tree(root.r, s)
    if root and not root.r and not root.l:
        s += " " + root.label
    elif root.label:
        s += "(" + root.label + sl + sr + ")"
    return s
def print_stack(s):
    ret = "["
    for t in s:
        ret += print_tree(t) + " ;;"
    ret += "]"
    return ret

# level order print for debugging
def traverse(root):
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
        if t1 and t2 and t1.label == t2.label:
            debug.write("label matched\n")
        if t1 is not None and t2 is not None:
            return (t1.label == t2.label) and \
            match_tree(t1.l, t2.l) and \
            match_tree(t1.r, t2.r)
        return False

    def binary_label_dfs(root, s0, s1): # subroutine for determining the label

        # debugging
        debug.write("\n$$$$$$$$$start_binary$$$$$$$$$\n")
        debug.write("root: " + print_tree(root) + "\n")
        debug.write("s0: " + print_tree(s0) + "\n")
        if root.l:
            debug.write("root.l: " + print_tree(root.l) + "\n---\n")
        debug.write("s1: " + print_tree(s1) + "\n")
        if root.r:
            debug.write("root.r: " + print_tree(root.r) + "\n")
        debug.write("\n*********end*********\n")

        if match_tree(root.l, s0) and match_tree(root.r, s1): # base case
            print("binary tree matched")
            return root.label
        if root.l:
            left_label = binary_label_dfs(root.l, s0, s1)
            return left_label if left_label else None
        if root.r and binary_label_dfs(root.r, s0, s1):
            right_label = binary_label_dfs(root.r, s0, s1)
            return right_label if right_label else None

    def unary_label_dfs(root, s0): # subroutine for determining the label

        # debugging
        debug.write("\n$$$$$$$$$start_unary$$$$$$$$$\n")
        debug.write("s0: " + print_tree(s0) + "\n")
        if root.l:
            debug.write("child: " + print_tree(root.l) + "\n")

        if match_tree(root.l, s0) and root.r == None: # base case
            print("unary tree matched")
            debug.write("match\n")
            debug.write("*********endsucc*********\n\n")
            return root.label
        debug.write("*********endfail*********\n\n")

        # recursive calls
        if root.l:
            child_label = unary_label_dfs(root.l, s0)
            if child_label:
                return child_label
        if root.r:
            child_label = unary_label_dfs(root.r, s0)
            return child_label if child_label else None

    actions = [] # returns actions list (seq of stack/buff)
    buff = list(map(Node, s.split()[::-1])) # reverse sentence for O(1) pop
    stack = []

    while buff or len(stack) > 1: # end when buffer consumed & stack has tree
        # try to reduce top two items
        if len(stack) > 1: # reduce
            right = stack[len(stack) - 2] # check this order
            left = stack[len(stack) - 1]
            new_node = Node(binary_label_dfs(t, left, right))
            if new_node.label: # found a matching reduce
                debug.write("~~~binary reduce~~~\n\n")
                new_node.l = stack.pop()
                new_node.r = stack.pop()
                stack.append(new_node)
                debug.write("stack: " + print_stack(stack) + "----------------------------------")
            else: # try to unary reduce
                child = stack[len(stack) - 1]
                new_node = Node(unary_label_dfs(t, left))
                if new_node.label: # found a unary reduce
                    debug.write("~~~unary reduce~~~\n\n")
                    new_node.l = stack.pop()
                    stack.append(new_node)
                    debug.write("stack: " + print_stack(stack)+ "----------------------------------")
                else: # shift
                    debug.write("~~~shift1~~~\n\n")
                    stack.append(buff.pop())
                    debug.write("stack: " + print_stack(stack)+ "----------------------------------")
        elif len(stack) == 1: # just try unary reduce
            child = stack[len(stack) - 1]
            new_node = Node(unary_label_dfs(t, child))
            if new_node.label: # found a unary reduce
                debug.write("~~~unary reduce~~~\n\n")
                new_node.l = stack.pop()
                stack.append(new_node)
                debug.write("stack: " + print_stack(stack)+ "----------------------------------")
            else: # shift
                debug.write("~~~shift2~~~\n\n")
                stack.append(buff.pop())
                debug.write("stack: " + print_stack(stack)+ "----------------------------------")
        else: # shift
            debug.write("~~~shift3~~~\n\n")
            stack.append(buff.pop())
            debug.write("stack: " + print_stack(stack))
        actions.append(stack + ["()"] + buff) # record action

    return actions


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
        tree_list.append(parse_tree(t[1:-1]))

    # use inorder traveral to generate sentences from trees
    sentences = []
    for t in tree_list:
        sentences.append(inorder_sentence(t).lstrip()) # extra space on left
    traverse(tree_list[0])
    # print(print_tree(tree_list[0]))

    with open(outpath + 'all.data', 'w') as f:
        for t, s in zip(tree_list, sentences):
            f.write('\n'.join(str(v) for v in generate_actions(t, s)))
            f.write("----------------------")
            break # test 1 tree

debug.close()
