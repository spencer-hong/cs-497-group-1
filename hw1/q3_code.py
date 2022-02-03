import numpy as np
import random
import csv
from q1and2_code import *

#shuffle data set, then split into training, test, and dev sets
def split_data(corpus):
    corpus.sort()
    random.seed(38)
    random.shuffle(corpus)
    
    train, test, dev = np.split(corpus, [int(.8*len(corpus)), int(.9*len(corpus))])

    with open("train.txt", "w", encoding="utf8") as train_file:
        csv.writer(train_file).writerows(train)
    train_file.close()

    with open("test.txt", "w", encoding="utf8") as test_file:
        csv.writer(test_file).writerows(test)
    test_file.close()
    
    with open("dev.txt", "w", encoding="utf8") as dev_file:
        csv.writer(dev_file).writerows(dev)
    dev_file.close()

if __name__ == '__main__':
    corpus = tokenize_and_tag('source_text.txt')
    split_data(corpus)