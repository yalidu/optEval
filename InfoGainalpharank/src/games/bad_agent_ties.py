import numpy as np
import logging


class BadAgentTies:
    
    def __init__(self, actions=100, seed=None, hardness=1):
        self.logger = logging.getLogger("BadAgentTies")
        self.actions = actions
        if self.actions == 2:
            p = 1 / hardness
            self.matrix = np.array([[
                [ 0,   p],
                [ -p,  0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        if self.actions == 3:
            p = 1 / hardness
            self.matrix = np.array([[
                [ 0,   p,  1],
                [ -p,  0,  p],
                [ -1, -p,  0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        self.matrix = np.load('/home/yali/yanxue/bernoullir2.npy')
        self.matrix = np.reshape(self.matrix, (1, actions, actions))
        self.Mmax = np.max(self.matrix)
        self.Mmin = np.min(self.matrix)

        self.matrix = (self.matrix - self.Mmin) / (self.Mmax - self.Mmin)
        #print(self.matrix)
        self.matrix = np.load('/home/yali/yanxue/bernoullir2.npy')
        self.matrix = np.reshape(self.matrix, (1, actions, actions))
        self.logger.debug("\n"+str(np.around(self.matrix, 2)))

    def get_entry_sample(self, entry):
        player1_win = np.random.binomial(10, self.matrix[0][tuple(entry)])
        return np.array([player1_win])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions