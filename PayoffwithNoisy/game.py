import numpy as np
import logging
import file_utils
import pickle



class Egame:

    def __init__(self,  name='Kuhn-poker',actions=10,seed=None, noise=1, clip=True):
        self.logger = logging.getLogger("GaussianGamesOnePop")
        self.actions = actions
        with open("/data/yanxue/spinning_top_payoffs.pkl", "rb") as fh:
            payoffs = pickle.load(fh)
        #self.matrix = np.random.random(size=(1, actions, actions))
        self.matrix=payoffs[name]
        mmax=np.max(self.matrix)
        mmin=np.min(self.matrix)
        self.matrix=(self.matrix-mmin)/(mmax-mmin)
        actions=self.matrix.shape[-1]
        self.actions=actions
        #print(self.matrix)
        self.matrix= np.reshape(self.matrix,(1,actions,actions))
        self.logger.debug("\n" + str(np.around(self.matrix, 2)))
    def get_entry_sample(self, entry):
        player1_win = np.random.binomial(10, p=self.matrix[0][tuple(entry)],size=1)
        #np.random.normal(self.matrix[0][tuple(entry)], size=1)

        return np.array([player1_win])

    def true_payoffs(self):
        #return self.matrix
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions

    '''
    def get_entry_sample(self, entry):
        wins=0
        for i in range(60):
            player1_win = np.random.normal(self.matrix[0][tuple(entry)], self.noise, size=1)
            if self.clip:
                mean_val = self.matrix[0][tuple(entry)]
                player1_win = np.clip(player1_win, mean_val - self.noise, mean_val + self.noise)
            wins+=player1_win
        return np.array([wins])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions
    '''

class GaussianOnePopGames:

    def __init__(self, actions=15,rank=5, seed=None, noise=1, clip=True):
        self.logger = logging.getLogger("GaussianGamesOnePop")
        self.actions = actions
        self.noise = noise
        self.clip = clip
        if seed is not None:
            np.random.seed(seed)
        #self.matrix = np.random.random(size=(1, actions, actions))

        if rank==2:
            self.matrix=np.load("gaussian15.npy")
        else:
            self.matrix = np.load("conv{}.npy".format(actions))  #
        self.matrix= np.reshape(self.matrix,(1,actions,actions))
        self.logger.debug("\n" + str(np.around(self.matrix, 2)))
        if seed is not None:
            np.random.seed()


    def get_entry_sample(self, entry):
        wins=0
        for i in range(10):
            player1_win = np.random.normal(self.matrix[0][tuple(entry)], self.noise, size=1)
            if self.clip:
                mean_val = self.matrix[0][tuple(entry)]
                player1_win = np.clip(player1_win, mean_val - self.noise, mean_val + self.noise)
            wins+=player1_win
        return np.array([wins])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions
class Bernoulli_game:
    def __init__(self, actions=10, seed=None):
        self.logger = logging.getLogger("Bernoulli_game")
        self.actions = actions


        self.matrix = self.get_soccer_data()
        self.matrix=np.reshape(self.matrix,(1,actions,actions))
        self.Mmax=np.max(self.matrix)
        self.Mmin=np.min(self.matrix)

        self.P=(self.matrix-self.Mmin)/(self.Mmax-self.Mmin)
        #np.random.random(size=(1, actions, actions))
        #self.P=self.matrix
        self.logger.debug("\n" + str(np.around(self.matrix, 2)))


    def get_soccer_data(self):
        """Returns the payoffs and strategy labels for MuJoCo soccer experiments."""
        payoff_file =np.load("bernoulli10.npy")#np.load("bernoulli10.npy")
        #payoffs=2*payoffs-1
        return payoff_file

    def get_entry_sample(self, entry):
        player1_win = np.random.binomial(10, p=self.P[0][tuple(entry)],size=1)
        #np.random.normal(self.matrix[0][tuple(entry)], size=1)

        return np.array([player1_win])

    def true_payoffs(self):
        #return self.matrix
        return self.P
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions
class Soccer_game:
    def __init__(self, actions=200, seed=None):
        self.logger = logging.getLogger("Soccer_game")
        self.actions = actions


        self.matrix = self.get_soccer_data()
        self.actions=self.matrix.shape[-1]
        self.matrix=np.reshape(self.matrix,(1,self.actions,self.actions))
        #print(self.matrix)

        #print(self.P)
        #print(self.P*( (self.Mmax - self.Mmin))+self.Mmin)
        #np.random.random(size=(1, actions, actions))
        self.logger.debug("\n" + str(np.around(self.matrix, 2)))


    def get_soccer_data(self):
        """Returns the payoffs and strategy labels for MuJoCo soccer experiments."""

        payoffs = np.load("soccer200.npy")
        #payoffs=2*payoffs-1
        #print(payoffs)
        return payoffs

    def get_entry_sample(self, entry):
        player1_win = np.random.binomial(10, p=self.matrix[0][tuple(entry)],size=1)
        #np.random.normal(self.matrix[0][tuple(entry)], size=1)

        return np.array([player1_win])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions


import numpy as np
import logging


class BadAgentTies:

    def __init__(self, actions=4, seed=None, hardness=1):
        self.logger = logging.getLogger("BadAgentTies")
        self.actions = actions
        if self.actions == 2:
            p = 1 / hardness
            self.matrix = np.array([[
                [0, p],
                [-p, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        if self.actions == 3:
            p = 1 / hardness
            self.matrix = np.array([[
                [0, p, 1],
                [-p, 0, p],
                [-1, -p, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions == 4:
            p = 1 / hardness
            self.matrix = np.array([[
                [0, -p, 1, 1],
                [p, 0, 1, 1],
                [-1, -1, 0, 0],
                [-1, -1, 0, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions == 5:
            p = 1 / hardness
            self.matrix = np.array([[
                [0, -p, p, 1, 1],
                [p, 0, -p, 1, 1],
                [-p, p, 0, 1, 1],
                [-1, -1, -1, 0, 0],
                [-1, -1, -1, 0, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions == 6:
            p = 1 / hardness
            self.matrix = np.array([[
                [0, -p, p, 1, 1, 1],
                [p, 0, -p, 1, 1, 1],
                [-p, p, 0, 1, 1, 1],
                [-1, -1, -1, 0, 0, 0],
                [-1, -1, -1, 0, 0, 0],
                [-1, -1, -1, 0, 0, 0],
            ]])
            self.matrix = (self.matrix + 1) / 2
        elif self.actions > 6:
            p = 1 / hardness
            self.matrix = np.ones(shape=(1, self.actions, self.actions))
            self.matrix[0, :3, :3] = np.array([
                [0, -p, p],
                [p, 0, -p],
                [-p, p, 0],
            ])
            self.matrix[0, 3:, :3] = -1
            self.matrix[0, :3, 3:] = 1
            self.matrix[0, 3:, 3:] = 0
            self.matrix = (self.matrix + 1) / 2
        print(self.matrix)
        self.logger.debug("\n" + str(np.around(self.matrix, 2)))

    def get_entry_sample(self, entry):
        player1_win = np.random.binomial(1, self.matrix[0][tuple(entry)])
        return np.array([player1_win])

    def true_payoffs(self):
        return self.matrix
        # return np.array([self.matrix])

    def get_env_info(self):
        # Return #populations, #players, #strats_per_player
        return 1, 2, self.actions

if __name__ == '__main__':
    print(11111)
    X=Egame()
    #print(X.true_payoffs())
    #print(X.get_entry_sample((0,1)))
    #print(np.clip(, mean_val - self.noise, mean_val + self.noise))