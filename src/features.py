from utils import *
import pickle

# feature list for one action:
# 0: ground truth action
#----------------------------------------
# 1-4: top four words on buffer, <null> if they don't exist
#----------------------------------------
# 5: label of stack[0] ("null" if word)
# 6: word of stack[0] ("null" if constituent label)
# 7-10: leftmost word and POS, rightmost word and POS
#----------------------------------------
# 11: label of stack[1] ("null" if word)
# 12: word of stack[1] ("null" if constituent label)
# 13-16: leftmost word and POS, rightmost word and POS
#----------------------------------------
# 17: label of stack[2] ("null" if word)
# 18: word of stack[2] ("null" if constituent label)
# 19-22: leftmost word and POS, rightmost word and POS
#----------------------------------------
# 23: label of stack[3] ("null" if word)
# 24: word of stack[3] ("null" if constituent label)
# 25-28: leftmost word and POS, rightmost word and POS

def rearrange(f):
    labels = set([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    word_start = 12

    new_list = [0] * 28
    for i in range(1, len(f)):
        if i in labels:
            new_list[i] = f[i]
        else:
            new_list[word_start] = f[i]
            word_start += 1

    new_list.append(f[0]) # append label to end

    return new_list



def get_left(t):
    if not t.l.r and not t.l.l: # if l child is unary
        return [t.label, unindex(t.l.label)]
    else:
        return get_left(t.l)

def get_right(t):
    if not t.l.r and not t.l.l: # if l child is unary
        return [t.label, unindex(t.l.label)]
    else:
        if t.r: # handle strange case of "(NP (NNP Moscow) ))"
            return get_right(t.r)
        else:
            return get_right(t.l)

def unindex(a):
    return a.split("/")[0].rstrip().lstrip() # assume no words contain "/"

if __name__ == '__main__':
    datapath = "../data/all.data"
    outpath = "../data/"

    # open file and save as one large string
    text = ""
    with open(datapath, 'r') as f:
        text += f.read().replace('\n', '')

    trees = text.split(tree_sep[1:-1])
    final_list = [] # list of lists of features
    for t in trees:
        if not t:
            continue
        actions = t.split(action_sep[1:-1])
        for a in actions:
            features = []
            if not a: continue# gets rid of weird empty string error
            items = a.split(sep[1:-1]) # remove surround \n
            features.append(items[0]) # features[0] is label
            stack = list(filter(None, items[1].split(list_sep)))
            buff = list(filter(None, items[2].split(list_sep)))

            # top four buffer words
            for i in range(0,4):
                if len(buff) > i:
                    features.append(unindex(buff[i]))
                else:
                    features.append("<null>")

            # stack items
            for i in range(0,4):
                if len(stack) > i:
                    tree = parse_tree(stack[i])
                    if not tree.l and not tree.r: # word
                        features.append(unindex(tree.label))
                        features.append("<null>")
                    else:
                        features.append("<null>")
                        features.append(tree.label)
                    if tree.l and tree.r: # binary rule
                        # assume a depth of 3 at least
                        features.extend(get_left(tree.l))
                        features.extend(get_right(tree.r))
                    else:
                        features.extend(["<null>"] * 4)
                else:
                    features.extend(["<null>"] * 5)
            final_list.append(rearrange(features))

    with open(outpath + "features.data", "wb") as f:
        pickle.dump(final_list, f)

    # write in readable form
    # i = 1
    # with open(outpath + "features_read.data", "w") as f:
    #     for fl in final_list:
    #         f.write(str(fl[0:1]) + "\n")
    #         f.write(str(fl[1:4]) + "\n")
    #         f.write(str(fl[5:10]) + "\n")
    #         f.write(str(fl[11:16]) + "\n")
    #         f.write(str(fl[17:22]) + "\n")
    #         f.write(str(fl[23:28]) + "\n")
    #         f.write(str(fl[23:28]) + "\n\n\n")
    #         i += 1
    #         if i == 500: break