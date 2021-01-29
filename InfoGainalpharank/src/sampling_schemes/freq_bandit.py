import numpy as np
import logging
from itertools import product
from copy import copy
import random


class FreqBandit:
    
    def __init__(self, num_pops, num_strats, num_players, max_payoff=1, min_payoff=0, delta=0.1, alpha_rank_func=None):
        self.num_pops = num_pops
        self.num_strats = num_strats
        self.num_players = num_players

        assert alpha_rank_func is not None
        self.alpha_rank = alpha_rank_func

        self.delta = delta
        self.range = max_payoff - min_payoff

        shape = (num_pops, *[num_strats for _ in range(num_players)])
        self.means = np.zeros(shape=shape)
        self.counts = np.zeros(shape=shape)

        self.logger = logging.getLogger("Freq_Bandit")
        np.random.seed()

        self.unresolved_pairs = set()
        if self.num_pops == 1:
            for i in range(num_strats):
                for j in range(num_strats):
                    if i == j:
                        continue
                    self.unresolved_pairs.add((
                        (i,j),
                        (j,i),
                        0
                    ))
        else:
            for base_strat in product(range(num_strats), repeat=num_players):
                for n in range(num_players):
                    # For each player that can deviate
                    for strat_index in range(num_strats):
                        # For each strategy they can change to 
                        if strat_index == base_strat[n]:
                            continue # Not a different strategy, move on
                        new_strat = copy(list(base_strat))
                        new_strat[n] = strat_index
                        new_strat = tuple(new_strat)
                        for p in range(num_pops):
                            # For each payoff entry in the #num_pop matrices for this deviation

                            # Unresolved pair is base_strat, new_strat, payoff_matrix
                            unresolved_pair = (base_strat, new_strat, p)
                            self.unresolved_pairs.add(unresolved_pair)

    def choose_entry_to_sample(self):
        if len(self.unresolved_pairs) == 0:
            return None, {}
        self.logger.debug("Unresolved pairs has {} elements".format(len(self.unresolved_pairs)))

        # Uniformly pick a an unresolved pair and uniformly pick a strategy
        pair = random.sample(self.unresolved_pairs, k=1)[0]
        strat = pair[random.randint(0, 1)]
        return strat, {}

    def update_entry(self, strats, payoffs):
        # Update the entries for the strategy
        for player, payoff in enumerate(payoffs):
            self.counts[player][tuple(strats)] += 1
            N = self.counts[player][tuple(strats)]
            self.means[player][tuple(strats)] = ((N - 1) * self.means[player][tuple(strats)] + payoff) / N

        # Update the unresolved strategy pairs
        # Brute force for now
        pairs_to_remove = set()
        for pair in self.unresolved_pairs:
            base_strat, new_strat, p = pair
            # Test if the confidence intervals don't overlap
            bm = self.means[p][tuple(base_strat)]
            bc = self.counts[p][tuple(base_strat)]
            nm = self.means[p][tuple(new_strat)]
            nc = self.counts[p][tuple(new_strat)]
            if bc == 0 or nc == 0:
                continue # We have no observed evaluations, CI overlaps
            bi = np.sqrt((np.log(2/self.delta) * (self.range**2))/(2*bc))
            base_lower = bm - bi
            base_upper = bm + bi
            ni = np.sqrt((np.log(2/self.delta) * (self.range**2))/(2*nc))
            new_lower = nm - ni
            new_upper = nm + ni

            if bm >= nm and new_upper > base_lower:
                pass
            elif bm < nm and base_upper > new_lower:
                pass
            else:
                # Remove the pair
                pairs_to_remove.add(pair)
        
        self.logger.debug("Removing {} elements. After seeing {} with payoffs {}".format(len(pairs_to_remove), strats, payoffs))
        for pair_to_remove in pairs_to_remove:
            self.unresolved_pairs.discard(pair_to_remove)

    def sample(self):
        # Sample from the distribution over payoff entries
        # Uniform over the confidence interval
        counts = np.maximum(self.counts, 1)
        interval = np.sqrt((np.log(2/self.delta) * (self.range**2))/(2*counts))

        min_vals = self.means - interval
        max_vals = self.means + interval

        sampled_values = np.random.rand(*self.means.shape) * (max_vals - min_vals) + min_vals

        return sampled_values

    def alpha_rankings_distrib(self, graph_samples=None, mean=False):
        # Return the alpha rank of the mean payoff matrix
        phi = self.alpha_rank(self.means)
        if mean:
            return np.array([phi])
        else:
            samples = [self.sample() for _ in range(graph_samples)]
            phis = [self.alpha_rank(p) for p in samples]
            return np.array(phis), np.array([phi])

    def payoff_distrib(self):
        return np.copy(self.means), np.zeros_like(self.means)