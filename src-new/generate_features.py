from utils import *
import os
import pickle
import time

def actions_to_features():
    t0 = time.time()
    datapath = "../data/actions/"
    outpath = "../data/features/"

    train_list = []
    test_list = []

    for file in os.listdir(datapath):
        print(file)
        if file.startswith('.'): continue
        # open file and save as one large string
        data_list = pickle.load(open(datapath + file, 'rb'))

        # list of lists of features
        # final_list_read = [] # list of lists of features for debugging

        final_list = []
        for d in data_list:
            features  = [remove_trailing(d.label)] + extract_features(d)
            final_list.append(rearrange(features))
#             final_list_read.append(features)

        if file == '23_actions.data':
            test_list.extend(final_list)
        else:
            train_list.extend(final_list)

    with open(outpath + "train.data", "wb") as f: pickle.dump(train_list, f)
    with open(outpath + "test.data", "wb") as f: pickle.dump(test_list, f)
    # with open(outpath + "validation.data", "wb") as f: pickle.dump(train_list[train_val_cut:], f)
    # with open(outpath + "validation.data")


#     # write in readable form
#     i = 1
#     with open(outpath + "features_read.txt", "w") as f:
#         for fl1, fl2 in zip(final_list, final_list_read):
#             f.write(str(fl1[:12]) + "\n")
#             f.write(str(fl1[12:]) + "\n\n")

#             f.write(str(fl2[0:1]) + "\n")
#             f.write(str(fl2[1:5]) + "\n")
#             f.write(str(fl2[5:11]) + "\n")
#             f.write(str(fl2[11:17]) + "\n")
#             f.write(str(fl2[17:23]) + "\n")
#             f.write(str(fl2[23:29]) + "\n")
#             i += 1
#             if i == 5000: break
    print("runtime: " + str(time.time() - t0))

if __name__ == "__main__":
    actions_to_features()
    vocab = Vocab("../data/mini/features/validation.data")
    with open('vocab.txt', 'w') as f:
        f.write("words: " + str(vocab.words) + "\n\n")
        f.write("actions: " + str(vocab.output_acts) + "\n\n")
        f.write("labels: " + str(vocab.feat_acts) + "\n\n")
