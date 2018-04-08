# feature list for one action:
# 0-3: top four words on buffer, <null> if they don't exist
# 4-8: label of stack[0], leftmost word and POS, rightmost word and POS
# 9-13: label of stack[1], leftmost word and POS, rightmost word and POS
# 14-18: label of stack[2], leftmost word and POS, rightmost word and POS
# 19-23: label of stack[3], leftmost word and POS, rightmost word and POS

if __name__ == '__main__':
    datapath = "../data/all.data"
    outpath = "../data/"

    # open file and save as one large string
    text = ""
    with open(datapath, 'r') as f:
        text += f.read().replace('\n', '')

    trees = text.split(tree_sep)
    for t in tree:
        actions = t.split(action_sep)
        for a in actions:

