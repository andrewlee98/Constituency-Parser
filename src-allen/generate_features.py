
from utils import *
import os
import pickle
import time

def actions_to_features():
    t0 = time.time()
    datapath = "data/actions/"
    outpath = "data/features/"
    examplepath = 'debug/sample_features.txt'
    example_count= 0

    train_list = []
    test_list = []

    for file in os.listdir(datapath):
        print(file)
        if file.startswith('.'): continue
        # if file != '00_actions.data': continue
        curr_file = file[0:2]
        data_list = pickle.load(open(datapath + file, 'rb'))

         # list of lists of features
        # final_list_read = [] # list of lists of features for debugging

        final_list = []
        for d in data_list:
            # keep the action dictionary consistent with the constituent label dictionary
            label =  d.label if 'shift' in d.label else d.label.split()[0] + ' '+ remove_trailing(d.label.split()[1])
            features  = [label] + extract_features(d)
            final_list.append(rearrange(features))

        # save some examples to debug file
        if example_count == 0:
            with open(examplepath, 'w') as f:
                for l in final_list: f.write(str(l) + '\n')
            example_count += 1

        with open(outpath + curr_file + '_features.data', "wb") as f: pickle.dump(final_list, f)


    # train_val_cut = int(8/10 * len(train_list))
    # val_test_cut =  int(9/10 * len(train_list))
    # with open(outpath + "test.data", "wb") as f: pickle.dump(train_list[train_val_cut:val_test_cut], f)
    # with open(outpath + "validation.data", "wb") as f: pickle.dump(train_list[val_test_cut:], f)
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
