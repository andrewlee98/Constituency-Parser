import dynet as dynet
import random
import numpy as np

class Network:
    def __init__(self, vocab, properties):
        self.properties = properties
        self.vocab = vocab

        # first initialize a computation graph container (or model)
        self.model = dynet.Model()

        # assign the algorithm for backpropagation updates
        self.updater = dynet.AdamTrainer(self.model)
        self.updater.learning_rate = 0.005

        # create embeddings for words and tag features.
        self.word_embedding = self.model.add_lookup_parameters((vocab.num_words(), properties.word_embed_dim))
        self.tag_embedding = self.model.add_lookup_parameters((vocab.num_tag_feats(), properties.pos_embed_dim))

        # assign transfer function
        self.transfer = dynet.rectify  # can be dynet.logistic or dynet.tanh as well

        # define the input dimension for the embedding layer
        # here we assume to see two words after and before and current word (meaning 5 word embeddings)
        # and to see the last two predicted tags (meaning two tag embeddings)
        self.input_dim = 15 * properties.word_embed_dim + 12 * properties.pos_embed_dim

        # define the hidden layer
        self.hidden_layer = self.model.add_parameters((properties.hidden_dim, self.input_dim))

        # define the hidden layer bias term and initialize it as constant 0.2
        self.hidden_layer_bias = self.model.add_parameters(properties.hidden_dim, init=dynet.ConstInitializer(0.2))

        # define the output weight
        self.output_layer = self.model.add_parameters((vocab.num_tags(), properties.hidden_dim))

        # define the bias vector and initialize it as zero
        self.output_bias = self.model.add_parameters(vocab.num_tags(), init=dynet.ConstInitializer(0))

    def build_graph(self, features):
        # extract word and tags ids
        word_ids = [self.vocab.word2id(word_feat) for word_feat in features[12:-1]]
        tag_ids = [self.vocab.feat_tag2id(tag_feat) for tag_feat in features[0:12]]

        # extract word embeddings and tag embeddings from features
        word_embeds = [self.word_embedding[wid] for wid in word_ids]
        tag_embeds = [self.tag_embedding[tid] for tid in tag_ids]

        # concatenating all features (recall that '+' for lists is equivalent to appending two lists)
        embedding_layer = dynet.concatenate(word_embeds + tag_embeds)

        # calculating the hidden layer
        # .expr() converts a parameter to a matrix expression in dynet (its a dynet-specific syntax)
        hidden = self.transfer(self.hidden_layer.expr() * embedding_layer + self.hidden_layer_bias.expr())

        # calculating the output layer
        output = self.output_layer.expr() * hidden + self.output_bias.expr()

        # return the output as a dynet vector (expression)
        return output
    
    
    
    
    def calc_acc(self, s):
        tot, corr = 0, 0
        for v in s:
            if self.decode(v[:-1]) == v[-1]: corr += 1
            tot += 1
        return corr/tot
    
    def train(self, train_file, epochs, validation_file):
        plot_on = True
        # matplotlib config
        loss_values = []
        validation_data = pickle.load(open(validation_file, 'rb'))
        validation_accs, train_accs = [], []
        

        
        train_data_original = pickle.load(open(train_file, "rb" ))
        
        for i in range(epochs):
            print('started epoch', (i+1))
            losses = []
            train_data = pickle.load(open(train_file, "rb" ))

            # shuffle the training data.
            random.shuffle(train_data)

            step = 0
            for fl in train_data:
                features, label = fl[:-1], fl[-1]
                gold_label = self.vocab.tag2id(label)
                result = self.build_graph(features)

                # getting loss with respect to negative log softmax function and the gold label
                loss = dynet.pickneglogsoftmax(result, gold_label)

                # appending to the minibatch losses
                losses.append(loss)
                step += 1

                if len(losses) >= self.properties.minibatch_size:
                    # now we have enough loss values to get loss for minibatch
                    minibatch_loss = dynet.esum(losses) / len(losses)

                    # calling dynet to run forward computation for all minibatch items
                    minibatch_loss.forward()

                    # getting float value of the loss for current minibatch
                    minibatch_loss_value = minibatch_loss.value()

                    # printing info and plotting
                    loss_values.append((len(loss_values), minibatch_loss_value))
                    if len(loss_values)%10==0:
                        


                            
                        progress = round(100 * float(step) / len(train_data), 2)
                        print('current minibatch loss', minibatch_loss_value, 'progress:', progress, '%')

                    # calling dynet to run backpropagation
                    minibatch_loss.backward()

                    # calling dynet to change parameter values with respect to current backpropagation
                    self.updater.update()

                    # empty the loss vector
                    losses = []

                    # refresh the memory of dynet
                    dynet.renew_cg()
                    
                    # get validation set accuracy
                    if len(loss_values)%100==0: 
                        validation_accs.append((len(loss_values), self.calc_acc(validation_data)))
                        train_accs.append((len(loss_values), self.calc_acc(train_data_original)))

            # there are still some minibatch items in the memory but they are smaller than the minibatch size
            # so we ask dynet to forget them
            dynet.renew_cg()
            
        # return these values just for plotting
        return loss_values, validation_accs, train_accs

    def decode(self, features):

        # running forward
        output = self.build_graph(features)

        # getting list value of the output
        scores = output.npvalue()

        # getting best tag
        best_tag_id = np.argmax(scores)

        # assigning the best tag
        pred = self.vocab.tagid2tag_str(best_tag_id)

        # refresh dynet memory (computation graph)
        dynet.renew_cg()

        return pred

    def load(self, filename):
        self.model.populate(filename)

    def save(self, filename):
        self.model.save(filename)
        
        
###############################

from collections import defaultdict
import pickle

class Node:
    def __init__(self, label):
        self.label = label
        self.l = None
        self.r = None

class datum:
    def __init__(self, stack, buff, label):
        self.stack = stack
        self.buff = buff
        self.label = label

# generate sentences from the tree
def inorder_sentence(root, s = ""):
    if not root.l and not root.r:
        s += " " + root.label
        return s
    if root.l:
        s = inorder_sentence(root.l, s)
    if root.r:
        s = inorder_sentence(root.r, s)
    return s

# util to debug
def tree_to_str(root, s = ""):
    sr = ""
    sl = ""
    if root.l:
        sl = tree_to_str(root.l, s)
    if root.r:
        sr = tree_to_str(root.r, s)
    if root and not root.r and not root.l:
        s += " " + clean(root.label)
    elif root.label:
        s += " (" + clean(root.label) + sl + sr + ")"
    return s

def stack_to_str(s):
    if not s: return '[]'
    ret = "["
    for t in s[:-1]:
        ret += tree_to_str(t) + ", "
    ret += tree_to_str(s[-1]) + "]"
    return ret

# level order print for debugging
def level_order(root):
    current_level = [root]
    while current_level:
        print(' '.join(str(node.label) for node in current_level))
        next_level = list()
        for n in current_level:
            if n.l:
                next_level.append(n.l)
            if n.r:
                next_level.append(n.r)
            current_level = next_level

def clean(s): # remove excess space
        s = s.rstrip().lstrip()
        return s

# method for transforming one string into a tree
def parse_tree(tree_str):
    tree_str = clean(tree_str)
    if tree_str[0] == "(": # remove surrounding parentheses
        tree_str = tree_str[1:-1]
    root = Node(tree_str.split()[0]) # set first word as root
    tree_str = tree_str.split()[1:] # remove first (root) word
    tree_str = ' '.join(tree_str) # convert back to string


    stack = [] # use to keep track of parentheses
    nested = False # boolean for if in nested statement
    children = [] # list of strings for children

    i = 0
    while i < len(tree_str): # collect children into list
        # nested parentheses case
        if tree_str[i] == "(":
            stack.append("(")
            if not nested: # save index of outermost left paren
                start_idx = i
            nested = True
        elif tree_str[i] == ")" and nested:
            stack.pop()
            if not stack:
                children.append(tree_str[start_idx:i + 1])
                nested = False

        # handle base case string
        elif tree_str[i] not in "() \n\t" and not nested:
            start_idx = i
            while tree_str[i] not in "() \n\t" and i < len(tree_str) - 1:
                i += 1
            children.append(tree_str[start_idx:i + 1])
        i += 1
    # now the children list is complete, recurse
    children = list(map(clean, children)) # clean whitespace from children
    if len(children) == 1:
        root.l = parse_tree(children[0])
    elif len(children) == 2:
        root.l = parse_tree(children[0])
        root.r = parse_tree(children[1])
    elif len(children) > 2: # binarize case
        root.l = parse_tree(children[0])
        binarize_str = "(" + root.label
        binarize_str += "_inner " if "_inner" not in binarize_str else " "
        binarize_str += ' '.join(children[1:]) + ")"
        root.r = parse_tree(binarize_str) # recursively binarize
    return root


class Vocab:
    def __init__(self, data_path):
        feature_list = pickle.load(open(data_path, "rb" ))
        word_count, actions, labels = defaultdict(int), set(), set()

        for feats in feature_list:
            actions.add(feats[-1])
            for word in feats[12:-1]:
                word_count[word] += 1
            for t in feats[:12]:
                labels.add(t)
        actions = list(actions)
        labels = list(labels)
        words = [word for word in word_count.keys() if word_count[word] > 1]

        self.words = ['<UNK>'] + words
        self.word_dict = {word: i for i, word in enumerate(self.words)}

        self.output_acts = list(actions)
        self.output_act_dict = {a: i for i, a in enumerate(self.output_acts)}

        self.feat_acts = list(labels) + ['<UNK>']
        self.feat_acts_dict = {a: i for i, a in enumerate(self.feat_acts)}

    def tagid2tag_str(self, id):
        return self.output_acts[id]

    def tag2id(self, tag):
        return self.output_act_dict[tag]

    def feat_tag2id(self, tag): # definitely need to figure out a way around UNK
        return self.feat_acts_dict[tag] if tag in self.feat_acts_dict else self.feat_acts_dict['<UNK>']

    def word2id(self, word):
        return self.word_dict[word] if word in self.word_dict else self.word_dict['<UNK>']

    def num_words(self):
        return len(self.words)

    def num_tag_feats(self):
        return len(self.feat_acts)

    def num_tags(self):
        return len(self.output_acts)

###########################################


class NetProperties:
    def __init__(self, word_embed_dim, pos_embed_dim, hidden_dim, minibatch_size):
        self.word_embed_dim = word_embed_dim
        self.pos_embed_dim = pos_embed_dim
        self.hidden_dim = hidden_dim
        self.minibatch_size = minibatch_size
        
        
#####################################

def rearrange(f):
    # 0(5): label of stack[0] ("<word>" if word)
    # 1(7): leftmost POS
    # 2(9): rightmost POS
    # 3(11): label of stack[1] ("<word>" if word)
    # 4(13): leftmost POS
    # 5(15): rightmost POS
    # 6(17): label of stack[2] ("<word>" if word)
    # 7(19): leftmost POS
    # 8(21): rightmost POS
    # 9(23): label of stack[3] ("<word>" if word)
    # 10(25): leftmost POS
    # 11(27): rightmost POS
    labels = set([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27])
    new_list = []
    for i in range(1, len(f)):
        if i in labels:
            new_list.append(f[i])
    for i in range(1, len(f)):
        if i not in labels: # and i not in prefix:
            new_list.append(f[i])
    new_list.append(f[0]) # append label to end
    return new_list


#########################################

def replace_if_num(s):
    def is_num(s1):
        return s1.replace(',','').replace('.','',1).isdigit()
    return "<num>" if is_num(s) else s

def get_left(t):
    if not t.l.r and not t.l.l: # if l child is unary
        return [t.label, replace_if_num(unindex(t.l.label))]
    else:
        return get_left(t.l)

def get_right(t):
    if not t.l.r and not t.l.l: # if l child is unary
        return [t.label, replace_if_num(unindex(t.l.label))]
    else:
        if t.r: # handle strange case of "(NP (NNP Moscow) ))"
            return get_right(t.r)
        else:
            return get_right(t.l)

def unindex(a):
    return a.split("/")[0].rstrip().lstrip() # assume no words contain "/"

#######################################

def extract_features(d):
    features = []
    stack = d.stack[::-1]
    buff = d.buff

    # top four buffer words
    for i in range(0,4):
        if len(buff) > i:
            features.append(replace_if_num(unindex(buff[i])))
        else:
            features.append("<null>")

        # stack items
    for i in range(0,4):
        if len(stack) > i:
            tree = parse_tree(stack[i])
            if tree.l or tree.r: # label
                features.append(remove_trailing(tree.label))
                features.append("<label>")
            else: # word
                features.append("<word>")
                features.append(replace_if_num(unindex(tree.label)))


            if tree.l and tree.r: # binary rule
                # assume a depth of 3 at least
                features.extend(get_left(tree.l))
                features.extend(get_right(tree.r))
            else:
                features.extend(["<null>"] * 4)
        else:
            features.extend(["<null>"] * 6)

    return features

def remove_trailing(label):
    if label[-1] == label[0] and label[0] == '-': return label[1:-1]
    return ((label.split("-")[0]).split('_')[0]).split('=')[0]