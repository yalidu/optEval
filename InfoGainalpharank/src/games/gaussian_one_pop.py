import numpy as np
import logging


class GaussianOnePopGames:
    
    def __init__(self, actions=15, seed=None, noise=1, clip=True):
        self.logger = logging.getLogger("GaussianGamesOnePop")
        self.actions = actions
        self.noise = noise
        self.clip = clip
        if seed is not None:
            np.random.seed(seed)
        self.matrix=np.load("../gaussian15.npy")
        self.matrix = np.reshape(self.matrix,(1, actions, actions))
        print(self.matrix)
        self.logger.debug("\n"+str(np.around(self.matrix, 2)))
        if seed is not None:
            np.random.seed()

    def get_entry_sample(self, entry):
        sum=0
        for i in range(60):
            player1_win = np.random.normal(self.matrix[0][tuple(entry)], self.noise, size=1)
            if self.clip:
                mean_val = self.matrix[0][tuple(entry)]
                player1_win = np.clip(player1_win, mean_val - self.noise, mean_val + self.noise)
            sum+=player1_win
        return np.array([sum])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions