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
        while right.label[-5:] == 'inner' and right.label.split('_')[0] == n_t.label.split('_')[0]:
            n_t.children.append(right.l)
            right = right.r
        n_t.children.append(right)

    n_t.children = [debinarize_tree(child) for child in n_t.children]

    return n_t

def n_tree_to_str(root, s = ""):
    # base case
    if not root.children: s += " " + clean(root.label)
    elif root.label:
        s += " (" + clean(root.label) + ' '.join([n_tree_to_str(c) for c in root.children]) + ")"
    return s

t = "( (S     (NP-SBJ-1       (NP         (NP (DT The) (NN group) (POS 's) )        (NN president) )      (, ,)       (NP (NNP Peter) (NNP Chrisanthopoulos) )      (, ,) )    (VP (VBD was) (RB n't)       (PP-LOC-PRD (IN in)         (NP (PRP$ his) (NN office) ))      (NP-TMP (NNP Friday) (NN afternoon) )      (S-PRP         (NP-SBJ (-NONE- *-1) )        (VP (TO to)           (VP (VB comment) ))))    (. .) ))"
# t = "(S (S-TPC-1 (NP-SBJ (NP (NNP Japan) (POS 's)) (NP-SBJ_inner (JJ wholesale) (NNS prices))) (S-TPC-1_inner (PP-TMP (IN in) (NP (NNP September))) (VP (VP (VBD rose) (VP_inner (NP-EXT (CD 3.3) (NN %)) (PP-DIR (IN from) (ADVP-TMP (NP (DT a) (NN year)) (RBR earlier))))) (VP_inner (CC and) (VP (VBD were) (ADVP-PRD (RB up) (ADVP-PRD_inner (NP (CD 0.4) (NN %)) (PP (IN from) (NP (DT the) (NP_inner (JJ previous) (NN month))))))))))) (S_inner (, ,) (S_inner (NP-SBJ (NP (DT the) (NNP Bank)) (PP (IN of) (NP (NNP Japan)))) (S_inner (VP (VBD announced) (VP_inner (SBAR (-NONE- 0) (S (-NONE- *T*-1))) (NP-TMP (NNP Friday)))) (. .)))))"
t = parse_tree(t[1:-1])

dt = debinarize_tree(t)

print(tree_to_str(t))
print()
print(n_tree_to_str(dt))
print()
print(tree_to_str(parse_tree(n_tree_to_str(dt))))