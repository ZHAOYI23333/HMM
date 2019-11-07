# CSE 5522 Lab3 HMM
# Author: Yi Zhao
# Prof: Eric Fosler-Lussier
# Ohio State University

import pandas as pd 
import numpy as np

# import csv file
emiss_csv = pd.read_csv("data/observationProbs.csv")
emiss_csv.drop(emiss_csv.columns[0], axis = 1, inplace = True)
trans_csv = pd.read_csv("data/transitionProbs.csv")
trans_csv.drop(trans_csv.columns[0], axis = 1, inplace = True)
test_csv = pd.read_csv("data/testData.csv")
test_csv.drop(test_csv.columns[0], axis = 1, inplace = True)

# initialize data: emisssion, transition, test
emission = np.array(emiss_csv, dtype=float)
transition = np.array(trans_csv, dtype=float) 
test = np.array(test_csv, dtype=float)

# calculate golden sequence via Viterbi Algorithm
def viterbi(obs, emission, transition):

    obs -= 1 # match observation to the index in emission table
    obs = obs[obs>=0] # get rid of dummy ends

    score = np.zeros((2, len(obs)))
    path = np.zeros((2, len(obs)))
    gold_seq = []

    emission = np.log(emission)
    transition = np.log(transition)
    score[:,0] = emission[int(obs[0]),:] + transition[0:2, 2]

    for i in range(1,len(obs)):
        for j in range(2):
            score_list = score[:,i-1] + emission[int(obs[i]),j] + transition[j,0:2]
            score[j,i] =np.max(score_list)
            path[j,i] = np.where(score_list == np.max(score_list))[0][0]

    score[:, len(obs)-1] = score[:, len(obs)-1] + transition[2, 0:2]

    gold_seq.append(int(np.where(score[:, len(obs)-1] == np.max(score[:, len(obs)-1]))[0][0]))
    for i in range(len(obs)-1, 0, -1):
        last_label = int(gold_seq[-1])
        gold_seq.append(path[last_label,i])

    gold_seq.reverse()
    return ["H" if x>0 else "C" for x in gold_seq]

# output golden sequence 
for i in range(test.shape[0]):
    print(viterbi(test[i,:], emission, transition))




