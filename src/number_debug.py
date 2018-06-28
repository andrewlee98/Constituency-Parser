import os

if __name__ == '__main__':
    treepath = "../treebank/treebank_3/parsed/mrg/wsj/"
    outpath = "../data/"

    # open file and save as one large string
    text = ""
    for folder in os.listdir(treepath):
        if folder.startswith('.'):
            continue
        for filename in os.listdir(treepath + folder):
            if filename.startswith('.'):
                continue
            with open(treepath + folder + "/" + filename, 'r') as f:
                text = f.read().replace('\n', '')
            if "259" in text:
                print(filename)