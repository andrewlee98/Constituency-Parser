import os

class Node:
    def __init__(self, label):
        self.label = label
        self.l = None
        self.r = None

treepath = "../treebank/treebank_3/parsed/mrg/atis/"
outpath = "../data/"

labels = set()
tree_list = []

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


# compile labels
for t in tree_string_list:
    for w in t.split():
        if w[0] == "(" and len(w) > 1:
            labels.add(w.strip("("))


# method for transforming one string into a tree
def parse_tree(tree_str):

    if tree_str[0] == "(":
        tree_str = tree_str[1:-1] # remove surrounding parentheses
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

    # print(root.label)
    # print("children: " + str(children))

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



# turn tree strings into tree_list
for t in tree_string_list:
    tree_list.append(parse_tree(t))

def inorder(root, s = ""):
    if not root.l and not root.r:
        s += " " + root.label
        return s
    if root.l:
        s = inorder(root.l, s)
    if root.r:
        s = inorder(root.r, s)
    return s

sentences = []
for t in tree_list:
    sentences.append(inorder(t))

print(sentences)

# print testing
# tree_dfs(parse_tree(tree_string_list[0]))
