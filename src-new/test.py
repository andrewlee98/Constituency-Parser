import pickle
import dynet
import os
import numpy as np
from utils import *


if __name__ == '__main__':
    (vocab, net_properties) = pickle.load(open('../data/mini/vocab_net.data', 'rb'))
    network = Network(vocab, net_properties)
    network.load('../data/mini/net.model')

    writer = open('../data/mini/predictions.data', 'w')
    feature_list = pickle.load(open( "../data/mini/features/test.data", "rb" ))

    correct = 0
    shiftstars, tps, fps, fns, shifts, tpss, fpss, fnss = 0, 0, 0, 0, 0, 0, 0, 0
    for feature_set in feature_list:
        pred = network.decode(feature_set[:-1])
        writer.write("ground truth: " + feature_set[-1] + ", prediction: " + pred + "\n\n")

        # count shift stars
        if pred == "shift star":
            if feature_set[-1] == "shift star":
                shiftstars += 1
                tpss += 1
            else:
                fpss += 1
        if feature_set[-1] == "shift star" and pred != "shift star":
            fnss += 1
            shiftstars += 1

        # count shifts
        if pred == "shift":
            if feature_set[-1] == "shift":
                shifts += 1
                tps += 1
            else:
                fps += 1
        if feature_set[-1] == "shift" and pred != "shift":
            fns += 1
            shifts += 1

        # total accuracy
        if pred == feature_set[-1]:
            correct += 1


    print("accuracy: " + str(float(correct)/len(feature_list)))

#     print("----------------------------------------")

#     precision_ss = tpss / (tpss + fpss)
#     recall_ss = tpss / (tpss + fnss)
#     print("star precision: " + str(precision_ss))
#     print("star recall: " + str(recall_ss))
#     print("total stars: " + str(shiftstars))
#     print("star F1: " + str( 2*(precision_ss * recall_ss) / (precision_ss + recall_ss) ))

#     print("----------------------------------------")

#     precision_s = tps / (tps + fps)
#     recall_s = tps / (tps + fns)
#     print("shift precision: " + str(precision_s))
#     print("shift recall: " + str(recall_s))
#     print("total shifts: " + str(shifts))
#     print("shift F1: " + str( 2*(precision_s * recall_s) / (precision_s + recall_s) ))

    writer.close()
    
    
def remove_star(s):
    s = s.split()
    s = list(filter(lambda x: '*' not in x, s))
    return ' '.join(s)

def action(b, s, p):
    error = None

    if p.split()[0] == 'shift':
        if len(p.split()) > 1 and p.split()[1] == 'star':
            s.append(Node('*'))
            
        # normal shift
        try: s.append(b.pop(0))
        except: error = 'pop on empty buffer'

    elif p.split()[0] == 'unary':
        n = Node(clean(p.split()[1]))
        try:
            n.l = s.pop()
            s.append(n)
        except: error = 'unary on empty stack'

    else: # p.split()[0] == 'binary':
        n = Node(clean(p.split()[1]))
        try:
            n.r, n.l = s.pop(), s.pop()
            s.append(n)
        except: error = 'binary on insufficient stack'

    return b, s, error


if __name__ == '__main__':
    # load the network
    (vocab, net_properties) = pickle.load(open('../data/mini/vocab_net.data', 'rb'))
    network = Network(vocab, net_properties)
    network.load('../data/mini/net.model')

    # open treebank for testing
    treepath = "../treebank/treebank_3/parsed/mrg/wsj/00"

    # open file and save as one large string
    text = ""
    for filename in os.listdir(treepath):
        if filename.startswith('.'):
            continue
        with open(treepath + "/" + filename, 'r') as f:
            text += f.read().replace('\n', '')

    tree_string_list = []
    s = []
    start = 0
    for i in range(len(text)):
        if text[i] == "(":
            s.append("(")
        elif text[i] == ")":
            s.pop()
            if not s:
                tree_string_list.append(text[start : i + 1])
                start = i + 1

    # turn tree strings into tree_list
    tree_list = []
    for t in tree_string_list:
        tree_list.append((parse_tree(t[1:-1])))

    # use inorder traveral to generate sentences from trees
    sentences = []
    for t in tree_list:
        sentences.append(remove_star(inorder_sentence(t).lstrip()))

    # testing

    with open('../data/mini/tree_pred.txt', 'w') as outfile, open('../data/mini/evalb.txt', 'w') as evalb:
        for s, t in zip(sentences, tree_list):
            s = [clean(x) for x in s.split()]
            
            #debug
#             print(' '.join(s) + '\n') # print sentence
            outfile.write(' '.join(s) + '\n\n')

            # construct tree
            buff = list(map(Node, s))
            stack = []
            infinite_loop_count = 0 # terminate after 100 moves
            printed_from_error = False
            while buff or len(stack) > 1: # end when buff consumed & stack has tree
                
                
                # cast to string and predict
                stack, buff = list(map(tree_to_str, stack)), list(map(tree_to_str, buff))
                try: f = extract_features(datum(stack, buff, None))
                except: 
#                     print('feature extraction error')
                    printed_from_error = True
                    break
                    

                pred = network.decode(rearrange([0] + f)[:-1])
                # outfile.write(str(f) + ' ' +  pred + '\n')

                # cast back to Node and complete action
                stack, buff = list(map(Node, stack)), list(map(Node, buff))
                buff, stack, error = action(buff, stack, pred)
                if error:
                    # outfile.write(error + '\n')
#                     print('Error: ' + error)
#                     print(stack_to_str(stack) + '\n')
                    outfile.write('Error: ' + error + '\n')
                    outfile.write(stack_to_str(stack) + '\n\n')
                    evalb.write(stack_to_str(stack) + '\n\n')
                    printed_from_error = True
                    break
#                 print(pred + '\n' + stack_to_str(stack) + '\n')
                outfile.write(pred + '\n' + stack_to_str(stack) + '\n\n')
                
                infinite_loop_count += 1
                if infinite_loop_count >= 150: 
#                     print('infinite loop error')
#                     print(stack_to_str(stack) + '\n')
                    outfile.write('infinite loop error' + '\n')
                    outfile.write(stack_to_str(stack) + '\n\n')
                    evalb.write(stack_to_str(stack) + '\n\n')
                    printed_from_error = True
                    break
                
#             if not printed_from_error: print(stack_to_str(stack) + '\n')
            if not printed_from_error: 
                outfile.write(stack_to_str(stack) + '\n\n')
                evalb.write(stack_to_str(stack) + '\n\n')
#             print('GROUND TRUTH:\n' + tree_to_str(t) + '\n')
#             print('-------------------end of sentence-----------------\n')
            outfile.write('GROUND TRUTH:\n' + tree_to_str(t) + '\n\n')
            outfile.write('-------------------end of sentence-----------------\n\n')



