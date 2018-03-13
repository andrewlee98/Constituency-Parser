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
    print("input string: " + tree_str)

    tree_str = tree_str[1:-1] # remove surrounding parentheses
    root = Node(tree_str.split()[0]) # set first word as root
    tree_str = tree_str.split()[1:] # remove first (root) word
    tree_str = ' '.join(tree_str) # convert back to string


    stack = [] # use to keep track of parentheses
    nested = False # boolean for if in nested statement
    left = False # boolean for if a left child has been called on
    for i in range(len(tree_str)):

        # nested parentheses case
        if tree_str[i] == "(":
            stack.append("(")
            if not nested:
                start_idx = i
            nested = True
        elif tree_str[i] == ")" and nested:
            stack.pop()
            if not stack:
                if not left:
                    left_end = i
                    root.l = parse_tree(tree_str[start_idx:left_end + 1])
                    left = True
                else:
                    root.r = parse_tree(tree_str[left_end:i + 1])
            nested = False

        # handle base case string
        elif tree_str[i].isalpha() and not nested:
            start_idx = 1
            while tree_str[i].isalpha() and i < len(tree_str) - 1:
                i += 1
            if not left:
                left_end = i
                root.l = Node(tree_str[start_idx:left_end + 1])
                left = True
                print("Left Node: " + root.l.label)
                i += 10
            else:
                root.r = Node(tree_str[left_end:i + 1])
                print("Right Node: " + root.r.label)

    print("-------------------------")
    return root



# turn tree strings into tree_list
# for t in tree_string_list:
#     tree_list.append(parse_tree(t))

parse_tree(tree_string_list[0][2:-1])

# print out by levels
def traverse(rootnode):
  thislevel = [rootnode]
  while thislevel:
    nextlevel = list()
    for n in thislevel:
      print(n.label),
      if n.l: nextlevel.append(n.l)
      if n.r: nextlevel.append(n.r)
    print()
    thislevel = nextlevel

def tree_dfs(root):
    print(root.label)
    if root.l:
        tree_dfs(root.l)
    if root.r:
        tree_dfs(root.r)
