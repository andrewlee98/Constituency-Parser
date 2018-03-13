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


# turn tree strings into tree_list
for t in tree_string_list:
    tree_list.append(parse_tree(t))

# transform string into tree
def parse_tree(tree_str):
    root = Node(tree_str.split()[0])

    # remove surrounding parentheses
    tree_str = tree_str[1:-1]

    # set first word as root
    root = Node(tree_str.split()[0])

    # remove first word
    tree_str = tree_str.split()[1:]

    # base case (single/two words)
    if tree_str[0] != "(":
        root.l = tree_str[0]
        if tree_str[1]:
            root.r = tree_str[1]
        return root

    # search for right child
    s = tree_st[tree_st.find("(")+1:tree_st.rfind(")")]
    root.l = parse_tree(s)

    # search for left child
    for word in tree_str:
        root.r = parse_tree(s)

    return root

# use DFS for each constituent


# print testing
for t in tree_string_list:
    print(t)
    print("********************************")

print(labels)
