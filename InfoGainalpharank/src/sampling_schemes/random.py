import numpy as np
import logging
from itertools import product
from copy import copy
import random

class RandomSampler:
    
    def __init__(self, num_pops, num_strats, num_players, alpha_rank_func=None):
        self.num_pops = num_pops
        self.num_strats = num_strats
        self.num_players = num_players

        assert alpha_rank_func is not None
        self.alpha_rank = alpha_rank_func

        shape = (num_pops, *[num_strats for _ in range(num_players)])
        self.means = np.zeros(shape=shape)
        self.counts = np.zeros(shape=shape)

        self.logger = logging.getLogger("Random Sampler")
        np.random.seed()

    def choose_entry_to_sample(self):
        # Pick any strat uniformly
        strat = tuple([np.random.randint(self.num_strats) for _ in range(self.num_players)])
        return strat, {}

    def update_entry(self, strats, payoffs):
        # Update the entries for the strategy
        for player, payoff in enumerate(payoffs):
            self.counts[player][tuple(strats)] += 1
            N = self.counts[player][tuple(strats)]
            self.means[player][tuple(strats)] = ((N - 1) * self.means[player][tuple(strats)] + payoff) / N

    def alpha_rankings_distrib(self, graph_samples=None, mean=False):
        # Return the alpha rank of the mean payoff matrix
        phi = self.alpha_rank(self.means)
        if mean:
            return np.array([phi])
        else:
            return np.array([phi]), np.array([phi])

    def payoff_distrib(self):
        return np.copy(self.means), np.zeros_like(self.means)