from utils import *

def null_eval(pred, gold):
    i, j = 0, 0
    correct, fn, fp = 0, 0, 0

    while i < len(pred) and j < len(gold):
        p, g = pred[i], gold[j]

        if 'NONE' in p.label and 'NONE' in g.label:
            if ('*T*' in p.l.label and '*T*' in g.l.label) or ('0' in p.l.label and '0' in g.l.label) or ('*' in p.l.label and '*' in g.l.label and 'T' not in p.l.label and 'T' not in g.l.label): 
                correct += 1
            i, j = i + 1, j + 1
                
        # normal word match
        elif p.label == g.label and p.l.label == g.l.label:
            i, j = i + 1, j + 1
        else: # one them has a NONE
            if 'NONE' in p.label:
                fp, i = fp + 1, i + 1
            else: #'NONE' in g.label
                if 'NONE' not in g.label: return None # happens for one singleton sentence
                fn, j = fn + 1, j + 1

    if correct == 0 and (fp == 0 or fn == 0): return None
    # ag_prec = correct/(correct + fp)
    # ag_rec = correct/(correct + fn)

    return correct, fn, fp


def null_ag_eval(pred, gold):
    i, j = 0, 0
    correct, fn, fp = 0, 0, 0

    while i < len(pred) and j < len(gold):
        p, g = pred[i], gold[j]

        if 'NONE' in p.label and 'NONE' in g.label:
            correct, i, j = correct + 1, i + 1, j + 1
        # normal word match
        elif p.label == g.label and p.l.label == g.l.label:
            i, j = i + 1, j + 1
        else: # one them has a NONE
            if 'NONE' in p.label:
                fp, i = fp + 1, i + 1
            else: #'NONE' in g.label
                if 'NONE' not in g.label: return None # happens for one singleton sentence
                fn, j = fn + 1, j + 1

    if correct == 0 and (fp == 0 or fn == 0): return None
    # ag_prec = correct/(correct + fp)
    # ag_rec = correct/(correct + fn)

    return correct, fn, fp

with open('final_outputs/comp_trees_with_nulls.txt', 'r') as f:
    tree_list = f.read().split('-------------------end of sentence-----------------')
    pairs = [list(filter(lambda x: len(x) > 0, p.split('\n'))) for p in tree_list] # remove empty lines
    pairs = pairs[:-1] # remove the last elements, which is empty

    total = [0, 0, 0]
    for pair in pairs:
        pred = inorder_buffer_gold(parse_tree(pair[0]))
        gold = inorder_buffer_gold(parse_tree(pair[1]))
        scores = null_ag_eval(pred, gold)

        if scores:
            total[0] += scores[0]
            total[1] += scores[1]
            total[2] += scores[2]
    ag_prec = total[0]/(total[0] + total[2])
    ag_rec = total[0]/(total[0] + total[1])

    print('agnostic to null el: ', ag_prec, ag_rec)


    total = [0, 0, 0]
    for pair in pairs:
        pred = inorder_buffer_gold(parse_tree(pair[0]))
        gold = inorder_buffer_gold(parse_tree(pair[1]))
        scores = null_eval(pred, gold)

        if scores:
            total[0] += scores[0]
            total[1] += scores[1]
            total[2] += scores[2]
    ag_prec = total[0]/(total[0] + total[2])
    ag_rec = total[0]/(total[0] + total[1])

    print('exact: ', ag_prec, ag_rec)
