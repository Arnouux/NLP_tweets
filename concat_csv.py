PATH_NEG = "twitter-datasets/train_neg_full.txt"
PATH_POS = "twitter-datasets/train_pos_full.txt"

FINAL = "twitter-datasets/both_full.tsv"
FINAL_TRAIN = "twitter-datasets/train_shuffled_full.tsv"
FINAL_TEST = "twitter-datasets/test_shuffled_full.tsv"

train = []
x = 0
with open(FINAL, "w") as f:
    with open(PATH_NEG, "r", encoding="utf-8") as f2:
        line = f2.readline()
        while line != "":
            #### preprocessing
            line = " ".join(line.split())
            ####
            try:
                f.write(f"{line[:-1]}\t0\n")
            except:
                pass
            line = f2.readline()
            x += 1
            if x % 100000 == 0:
                print(x)
    with open(PATH_POS, "r") as f2:
        line = f2.readline()
        while line != "":
            #### preprocessing
            line = " ".join(line.split())
            ####
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
with open(FINAL, "r") as f:
    lines = f.readlines()
    random.shuffle(lines)
    with open(FINAL_TRAIN, "w", encoding="utf-8") as f2:
        for line in lines[:int(len(lines)*0.8)]:
            f2.write(f"{line[:-1]}\t0\n")

    with open(FINAL_TEST, "w", encoding="utf-8") as f2:
        for line in lines[int(len(lines)*0.8):]:
            f2.write(f"{line[:-1]}\t0\n")