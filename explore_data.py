PATH_NEG = "twitter-datasets/train_neg_full.txt"
PATH_POS = "twitter-datasets/train_pos_full.txt"

FINAL = "twitter-datasets/both.tsv"
FINAL_CHANGED = "twitter-datasets/both_zipf_luft.tsv"
FINAL_TRAIN_CHANGED = "twitter-datasets/train_shuffled_zipf_luft.tsv"
FINAL_TEST_CHANGED = "twitter-datasets/test_shuffled_zipf_luft.tsv"



words = {}
lines = []
x=0
with open(FINAL, "r", encoding="utf-8") as f:
    line = f.readline()
    while line != "":
        for w in line[:-3].split():
            
            if w in words:
                words[w] += 1
            else:
                words[w] = 1
        lines.append(line)
        line = f.readline()
        x+=1
        if x % 100000 == 0:
            print(x)
            
sort_words = sorted(words.items(), key=lambda x: x[1], reverse=True)
maxOcc = sort_words[0][1]*0.1
for i in sort_words[:10]:
	print(i[0], i[1])
 
res = [[ i for i, j in sort_words ],
       [ j for i, j in sort_words ]]


with open(FINAL_CHANGED, "w", encoding="utf-8") as f:
    for line in lines:
        end = line[-3:]
        line = line.split()
        new_line = []
        for word in line[:-1]:
            if word in words and words[word] < maxOcc and words[word] > 1 :
                new_line.append(word)
        if len(new_line) > 0:
            f.write(" ".join(new_line) + end)
        
        
x= 0
import random
with open(FINAL_CHANGED, "r", encoding="utf-8") as f:
    lines = f.readlines()
    random.shuffle(lines)
    with open(FINAL_TRAIN_CHANGED, "w", encoding="utf-8") as f2:
        for line in lines[:int(len(lines)*0.8)]:
            f2.write(f"{line[:-1]}\n")

    with open(FINAL_TEST_CHANGED, "w", encoding="utf-8") as f2:
        for line in lines[int(len(lines)*0.8):]:
            f2.write(f"{line[:-1]}\n")