from utils import *

with open('final_outputs/comp_trees_with_nulls.txt', 'r') as f:
    tree_list = f.read().split('-------------------end of sentence-----------------')
    pairs = [list(filter(lambda x: len(x) > 0, p.split('\n'))) for p in tree_list] # remove empty lines
    pairs = pairs[:-1] # remove the last elements, which is empty

    for pair in pairs:
        pred = parse_tree(pair[0])
        gold = parse_tree(pair[1])
