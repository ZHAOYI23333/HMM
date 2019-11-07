# CSE 5522 Lab3 HMM
# Author: Yi Zhao
# Prof: Eric Fosler-Lussier
# Ohio State University

import sys
import pandas as pd 
import numpy as np
import random as rd
import ast

# import csv file
emiss_csv = pd.read_csv("data/observationProbs.csv")
emiss_csv.drop(emiss_csv.columns[0], axis = 1, inplace = True)
trans_csv = pd.read_csv("data/transitionProbs.csv")
trans_csv.drop(trans_csv.columns[0], axis = 1, inplace = True)
test_csv = pd.read_csv("data/testData.csv")
test_csv.drop(test_csv.columns[0], axis = 1, inplace = True)

# initialize data: emisssion, transition, test
emission = np.array(emiss_csv, dtype=float)
transition = np.array(trans_csv, dtype=float)[0:2,:]
transition = transition / np.sum(transition, axis = 0) 
test = np.array(test_csv, dtype=float)

viterbi = [
    ['H', 'H', 'H', 'H', 'H'],
    ['H', 'H', 'H', 'H'],
    ['C', 'C', 'C', 'C', 'C'],
    ['C', 'C', 'C'],
    ['C', 'C', 'C', 'C', 'C'],
    ['C', 'C', 'C', 'C'],
    ['H', 'H', 'H'],
    ['H', 'H', 'C', 'C'],
    ['C', 'H', 'H', 'H', 'H'],
    ['C', 'C', 'C']]

def likelyhood(obs, emission, transition, res_viterbi):
    obs -= 1 # match observation to the index in emission table
    obs = obs[obs>=0] # get rid of dummy ends
    # print(obs)

    dict = {}
    hist = 10
    prev = 0
    for i in range(10000):
        seq = [2]
        weight = 1
        for j in range(len(obs)):
            num = rd.random()
            last = seq[-1]
            cur = int(num > transition[0, last])
            seq.append(cur)
            weight *= emission[int(obs[j]), cur]

        output = ["H" if x>0 else "C" for x in seq[1:]]
        key = str(output)
        dict[key] = weight if key not in dict else dict[key]+weight

        max_value = max(dict.values()) 
        max_key = [k for k, v in dict.items() if v == max_value][0]
        if (ast.literal_eval(max_key) == res_viterbi and max_key == prev):
            hist -= 1
            if hist <= 0:
                print("Converge at: ", i)
                return max_key
        prev = max_key
    max_value = max(dict.values()) 
    max_keys = [k for k, v in dict.items() if v == max_value]
    # converge if the gold sequence equals to the viterbi for 10 continuing samplings
    return max_keys[0]


# output golden sequence 
for i in range(test.shape[0]):
    print(likelyhood(test[i,:], emission, transition, viterbi[i]))
