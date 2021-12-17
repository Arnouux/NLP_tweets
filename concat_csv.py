PATH_NEG = "twitter-datasets/train_neg.txt"
PATH_POS = "twitter-datasets/train_pos.txt"

FINAL = "twitter-datasets/both.tsv"
FINAL_TRAIN = "twitter-datasets/train_shuffled.tsv"
FINAL_TEST = "twitter-datasets/test_shuffled.tsv"

import re
from nltk.metrics.distance import jaccard_distance
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer
from nltk.util import ngrams
import random

import nltk

correct_words = words.words()
stop_words = stopwords.words()
ps = PorterStemmer()

def clean_sentence(sentence):
    sentence = word_tokenize(sentence)
    count = 0
    for word in sentence:
        try:
            words_list = [
                (jaccard_distance(set(ngrams(word, 2)), set(ngrams(w, 2))),w)
                for w in correct_words if w[0]==word[0]
            ]
            word_best = sorted(words_list, key = lambda val:val[0])[0][1]
            count += 1
        except:
            word_best = word
        sentence[sentence.index(word)] = word_best
    return " ".join(sentence)
        

train = []
x = 0
with open(FINAL, "w", encoding="utf-8") as f:
    with open(PATH_NEG, "r", encoding="utf-8") as f2:
        line = f2.readline()
        while line != "":
            #### preprocessing
            line = line.replace("<user>", "")
            line = line.replace("<url>", "")
            line = " ".join(line.split())
            #line = clean_sentence(line)
            ####
            if len(line) > 1:
                try:
                    f.write(f"{line[:-1]}\t0\n")
                except:
                    pass
            line = f2.readline()
            x += 1
            if x % 10 == 0:
                print(x)
    with open(PATH_POS, "r", encoding="utf-8") as f2:
        line = f2.readline()
        while line != "":
            #### preprocessing
            line = line.replace("<user>", "")
            line = line.replace("<url>", "")
            line = " ".join(line.split())
            #line = clean_sentence(line)
            ####
            if len(line) > 1:
                try:
                    f.write(f"{line[:-1]}\t1\n")
                except:
                    pass
            line = f2.readline()
            x += 1
            if x % 100000 == 0:
                print(x)
        
    print("file created")

x= 0
import random
with open(FINAL, "r", encoding="utf-8") as f:
    lines = f.readlines()
    random.shuffle(lines)
    with open(FINAL_TRAIN, "w", encoding="utf-8") as f2:
        for line in lines[:int(len(lines)*0.8)]:
            f2.write(f"{line[:-1]}\n")

    with open(FINAL_TEST, "w", encoding="utf-8") as f2:
        for line in lines[int(len(lines)*0.8):]:
            f2.write(f"{line[:-1]}\n")