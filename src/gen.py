import os

treepath = "../treebank/treebank_3/parsed/prd/atis/"
outpath = "../outputs/"

for filename in os.listdir(treepath):
    with open(treepath + filename, 'r') as f:
        for line in f:
            for word in line.split():
                print(word)
