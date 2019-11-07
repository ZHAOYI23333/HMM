# CSE 5522, Adv.AI, Hidden Markov Model
# Author: Yi Zhao (zhao.2175@osu.edu)
# Professor: Eric Fosler-Lussier
# Ohio State University

import numpy as np

class HMM(object):

    def __init__(self, transition, emission, test):
        self.transition = np.array(transition)
        self.emission = np.array(emission)
        self.test = np.array(test)

    def viterbi(self):
        n = self.emission.shape[1] # the number of hidden variables classes
        obs = self.test - 1 # match observation to the index in emission table
        obs = obs[obs>=0] # get rid of dummy ends

        # initialize score and path table
        score = np.zeros((n, len(obs)))
        path = np.zeros((n, len(obs)))
        gold_seq = []

        # switch to log space to prevent underflow
        emission = np.log(self.emission)
        transition = np.log(self.transition)
        score[:,0] = self.emission[int(obs[0]),:] + self.transition[:-1, n]

        # dynamically calcuate best score 
        for i in range(1,len(obs)):
            for j in range(n):
                score_list = score[:,i-1] + emission[int(obs[i]),j] + transition[j,:-1]
                score[j,i] =np.max(score_list)
                path[j,i] = np.where(score_list == np.max(score_list))[0][0]
        # include end tag to the last observation
        score[:, len(obs)-1] = score[:, len(obs)-1] + transition[n, :-1]

        # trace back and find best score
        gold_seq.append(int(np.where(score[:, len(obs)-1] == np.max(score[:, len(obs)-1]))[0][0]))
        for i in range(len(obs)-1, 0, -1):
            last_label = int(gold_seq[-1])
            gold_seq.append(path[last_label,i])

        gold_seq.reverse()
        if n == 2:
            return ["H" if x>0 else "C" for x in gold_seq]
        else:
            return ["H" if x>1 else "C" if x<1 else "W" for x in gold_seq]

    def forward_backward(self):
        obs = self.test - 1 # switch observed sequence to index
        n = self.emission.shape[1] # number of class of tags. In lab, there would be 2 or 3.
        
        # initialize forward and backward table 
        forward = np.zeros((n, len(obs)))
        backward = np.zeros((n, len(obs)))

        forward[:,0] = self.emission[int(obs[0]),:] * self.transition[:-1, n]
        backward[:,-1] = self.transition[n, :-1] 

        # iteratively fill in forward table and backward table
        for i in range(1,len(obs)):
            for j in range(n):
                score_list = forward[:,i-1] * self.emission[int(obs[i]),j] * self.transition[j,:-1]
                forward[j,i] =np.sum(score_list)

        for i in range(len(obs)-2,-1,-1):
            for j in range(n):
                score_list = backward[:,i+1] * self.emission[int(obs[i+1]),:] * self.transition[:-1,j]
                backward[j,i] =np.sum(score_list)

        # update emission probability
        forwardBackwardProb = forward * backward
        fb_total = forward * backward
        fb_prob = fb_total / np.sum(fb_total, axis = 0)
        temp_emission = [np.sum(fb_prob[:,obs==0], axis = 1), 
            np.sum(fb_prob[:,obs==1], axis = 1),
            np.sum(fb_prob[:,obs==2], axis = 1)] / np.sum(fb_prob, axis = 1)

        # update transition probability
        for i in range(n):
            prior = forward[:,:-1] * backward[i,1:] * self.emission[obs[1:],i] * self.transition[i,:-1].reshape(n,-1)/ np.sum(fb_total[:,1:], axis = 0)
            self.transition[i,:-1] = np.sum(prior, axis = 1) / np.sum(fb_prob, axis = 1)
        self.transition[n,:-1] = fb_prob[:,-1] / np.sum(fb_prob, axis = 1)
        self.transition[:-1,n] = fb_prob[:,0]
        self.emission = temp_emission

    def train(self, iteration):
        for i in range(iteration+1):
            self.forward_backward()
        print(self.transition)
        print(self.emission)
        return self.viterbi()

if __name__ == "__main__":
    # Please give a transition table and emission table. 

    transition = [[0.25,0.25,0.25,0.2],[0.25,0.25,0.25,0.4],[0.25,0.25,0.25,0.4],[0.25,0.25,0.25,0]]
    emission = [[0.3,0.3,0.3],[0.3,0.3,0.3],[0.4,0.4,0.4]]

    # Here is the provided data in Excel. Could be used to examine correctness of implementation
    # transition = [[0.8,0.1,0.5],[0.1,0.8,0.5],[0.1,0.1,0]]
    # emission = [[0.7,0.1],[0.2,0.2],[0.1,0.7]]

    obs = [2,3,3,2,3,2,3,2,2,3,1,3,3,1,1,1,2,1,1,1,3,1,2,1,1,1,2,3,3,2,3,2,2]

    model = HMM(transition, emission, obs)
    print(model.train(100))