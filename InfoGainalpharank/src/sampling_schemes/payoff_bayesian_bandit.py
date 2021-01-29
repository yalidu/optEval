import numpy as np
import logging

class PayoffBayesianBandit:
    
    def __init__(self, 
                num_pops, 
                num_strats, 
                num_players=2, 
                payoff_distrib=None, 
                alpha_rank_func=None):

        self.num_strats = num_strats
        self.num_players = num_players
        self.num_pops = num_pops # num_pops != num_players always. For single population we have num_pops=1 and num_players=2 (or more in general)

        assert alpha_rank_func is not None
        self.alpha_rank = alpha_rank_func

        self.payoffs = payoff_distrib

        self.logger = logging.getLogger("Payoff_Bayesian_Bandit")

    def choose_entry_to_sample(self):
       max_strat = self.payoffs.info_gain_entry()
       return tuple(max_strat), {}

    def update_entry(self, strats, payoffs):
        self.payoffs.update_entry(strats, payoffs)

    def alpha_rankings_distrib(self, graph_samples=None, mean=False):
        if mean:
            mean_payoffs = self.payoffs.sample_mean()
            phi = self.alpha_rank(mean_payoffs)
            return np.array([phi])
        else:
            # For debugging and logging approximate the distribution over alpha-rankings
            samples = [self.payoffs.sample() for _ in range(graph_samples)]
            phis = list(map(self.alpha_rank, samples)) # To keep the same code for both

            # Also return the most probable alpha rank and its estimated probability
            unique_phis, counts = np.unique(np.around(phis, decimals=5), return_counts=True, axis=0)
            argmax_entry = np.argmax(counts)
            p_phi = unique_phis[argmax_entry]
            p_phi_prob = counts[argmax_entry] / sum(counts)

            return np.array(list(phis)), (np.array([p_phi]), p_phi_prob)

    def payoff_distrib(self):
        return self.payoffs.stats()
