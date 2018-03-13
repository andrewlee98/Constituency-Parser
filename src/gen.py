import os

class Node:
    def __init__(self, label):
        self.label = label
        self.l = None
        self.r = None

treepath = "../treebank/treebank_3/parsed/mrg/atis/"
outpath = "../data/"

labels = set()
text_list = []
buffer = ""

for filename in os.listdir(treepath):
    with open(treepath + filename, 'r') as f:
        text = f.read().replace('\n', '')

text_list = text.split("( END_OF_TEXT_UNIT )")
tree_text_list = []
for t in text_list:
    if "@" not in t and len(t) != 0:
        tree_text_list.append(t)

for t in tree_text_list:
    for w in t.split():
        if w[0] == "(" and len(w) > 1:
            labels.add(w.strip("("))


#
# def rec_parse(tree_str):
#     root = Node(string[0])
#     # remove surrounding parentheses
#
#     # unary case
#     if word
#
#     # search for right child
#     s = tree_st[tree_st.find("(")+1:tree_st.rfind(")")]
#     root.l = rec_parse(s)
#
#     # search for left child
#     for word in tree_str:
#         root.r = rec_parse(s)
#
#     return root

for t in tree_text_list:
    print(t)
    print("**********")

print(labels)
