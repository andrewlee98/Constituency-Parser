class Node:
    def __init__(self, label):
        self.label = label
        self.l = None
        self.r = None

tree_sep = "\n" + "*" * 24 + "\n" # denotes end of one tree's action sequence
action_sep = "\n" + "-" * 24 + "\n" # separates actions from each other
sep = "\n" + "=" * 24 + "\n" # separates action, stack, and buffer in one action
list_sep = ";;" # separates items in stack/buffer


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
