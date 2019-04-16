from utils import *


class n_Node:
    def __init__(self, label):
        self.label = label
        self.children = []


def debinarize_tree(b_t):
    n_t = n_Node(b_t.label)
    if b_t.l: n_t.children.append(b_t.l)

    if b_t.r:
        right = b_t.r
        while right.label[-5:] == 'inner':
            n_t.children.append(right.l)
            right = right.r
        n_t.children.append(right)

    n_t.children = [debinarize_tree(child) for child in n_t.children]

    return n_t

def n_tree_to_str(root, s = ""):
    # base case
    if not root.children: s += " " + clean(root.label)
    elif root.label:
        s += " (" + clean(root.label) + str([n_tree_to_str(c) for c in root.children]) + ")"
    return s

t = "( (S     (NP-SBJ-1       (NP         (NP (DT The) (NN group) (POS 's) )        (NN president) )      (, ,)       (NP (NNP Peter) (NNP Chrisanthopoulos) )      (, ,) )    (VP (VBD was) (RB n't)       (PP-LOC-PRD (IN in)         (NP (PRP$ his) (NN office) ))      (NP-TMP (NNP Friday) (NN afternoon) )      (S-PRP         (NP-SBJ (-NONE- *-1) )        (VP (TO to)           (VP (VB comment) ))))    (. .) ))"

t = parse_tree(t)

t = debinarize_tree(t)

print(n_tree_to_str(t))