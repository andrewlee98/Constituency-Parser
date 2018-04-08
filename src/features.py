if __name__ == '__main__':
    datapath = "../data/all.data"
    outpath = "../data/"

    # # open file and save as one large string
    # text = ""
    # test_file = os.listdir(treepath)[0]
    # print("testing: " + test_file + "/" + os.listdir(treepath + test_file)[0])
    # for folder in os.listdir(treepath):
    #     if folder.startswith('.'):
    #         continue
    #     for filename in os.listdir(treepath + folder):
    #         if filename.startswith('.'):
    #             continue
    #         with open(treepath + folder + "/" + filename, 'r') as f:
    #             text += f.read().replace('\n', '')
    #     break # test only one folder for speed