import numpy as np
import logging
from functools import partial
from scipy.stats import norm
import random


"""
This class implements a distribution over the payoff matrix's expected values.
Each entry is modelled independently as a Gaussian, with prior mu and var specified.
The data is assumed to be drawn from a Gaussian with unknown mean and variance 'noise_var'.
"""


class IndependentNormal:

    def __init__(self, num_pops, num_strats, num_players, starting_mu=0.5, starting_var=1, hallucination_samples=1, noise_var=1, estimate_noise=False):
        self.num_strats = num_strats
        self.num_players = num_players
        self.num_pops = num_pops # num_pops != num_players always. For single population we have num_pops=1 and num_players=2 (or more in general)

        shape = (num_pops, *[num_strats for _ in range(num_players)])
        self.means = np.ones(shape=shape) * starting_mu
        self.var = np.ones(shape=shape) * starting_var

        self.estimate_noise = estimate_noise
        self.noise_var_estimate = np.ones(shape=shape) * noise_var

        self.counts = np.zeros(shape=shape) # Number of data points seen
        self.running_total = np.zeros(shape=shape) # Sum of data points to make calculations easier
        self.running_total_sq = np.zeros(shape=shape) # Sum of data points squared to make calculations easier

        self.hallucination_samples = hallucination_samples
        self.starting_var = starting_var
        self.starting_mu = starting_mu
        self.noise_var = self.noise_var_estimate

        self.logger = logging.getLogger("Indep_Normal")
        self.logger_count = 0

    def update_entry(self, strats, payoffs):
        # Update means and sigmas
        k=60
        for player, payoff in enumerate(payoffs):
            # Update the payoff matrix for player p
            self.counts[player][tuple(strats)] += k
            self.running_total[player][tuple(strats)] += payoff
            self.running_total_sq[player][tuple(strats)] += payoff**2

            N = self.counts[player][tuple(strats)]
            sum_x = self.running_total[player][tuple(strats)]
            sum_x_sq = self.running_total_sq[player][tuple(strats)]

            if self.estimate_noise and N > 5:
                self.noise_var_estimate[player][tuple(strats)] = max((sum_x_sq - (2*sum_x*sum_x)/(N) + N*(sum_x/N)**2)/(N-1), 1e-5)

            nvr = self.noise_var_estimate[player][tuple(strats)]
            self.means[player][tuple(strats)] = (nvr * self.starting_mu + self.starting_var * sum_x)/(nvr + N * self.starting_var)
            self.var[player][tuple(strats)] = (nvr * self.starting_var) / (nvr + N * self.starting_var)

        if self.logger_count % 100 == 0:
            self.logger.debug("Means:")
            self.logger.debug("\n" + str(np.around(self.means, decimals=2)))

            self.logger.debug("Vars:")
            self.logger.debug("\n" + str(np.around(self.var, decimals=3)))
        self.logger_count += 1

    def info_gain_entry(self):
        # Return the entry to sample to maximise the expected information gain between payoff distribution and sample.
        # For independent Gaussian entries this is equivalent to picking the entry with the lowest number of samples
        possible_strats = np.argwhere(self.counts[0] == self.counts[0].min())
        return random.choice(possible_strats)

    def sample(self):
        return np.random.normal(self.means, np.sqrt(self.var))

    def prob(self, m_sample):
        # return norm(self.means, np.sqrt(1/self.counts)).pdf(m_sample).prod()
        probs = norm(self.means, np.sqrt(self.var)).pdf(m_sample)
        return np.log(probs.clip(min=1e-100)).sum()# Return log prob
    
    def sample_prob(self):
        # Sample and return the probability of that sample
        m_sample = self.sample()
        m_prob = self.prob(m_sample)
        return m_sample, m_prob
    
    def sample_mean(self):
        return np.copy(self.means)

    def hallucinate_sample_func(self, hallucinate_mean=True):
        # Hallucinate a sample (either the mean or a sampled element) and return the resulting payoff matrix
        return partial(h_sample_func, 
                        running_total=self.running_total,
                        counts=self.counts, 
                        num_pops=self.num_pops, 
                        hallucinate_samples=self.hallucination_samples,
                        starting_mu=self.starting_mu,
                        starting_var=self.starting_var,
                        noise_var=self.noise_var,
                        halluc_sample=self.sample_mean() if hallucinate_mean else self.sample() + np.random.normal(np.zeros_like(self.means), self.noise_var))

    def hallucinate_sample_prob_func(self, hallucinate_mean=True):
        # Hallucinate and return the probability of the payoff matrix under that new distribution
        return partial(h_sample_prob_func, 
                        running_total=self.running_total,
                        counts=self.counts, 
                        num_pops=self.num_pops, 
                        hallucinate_samples=self.hallucination_samples,
                        starting_mu=self.starting_mu,
                        starting_var=self.starting_var,
                        noise_var=self.noise_var,
                        halluc_sample=self.sample_mean() if hallucinate_mean else self.sample())

    def hallucinate_prob_func(self, hallucinate_mean=True):
        # Return probability of a sample under the hallucinated distribution
        return partial(h_prob_func, 
                        running_total=self.running_total,
                        counts=self.counts, 
                        num_pops=self.num_pops, 
                        hallucinate_samples=self.hallucination_samples,
                        starting_mu=self.starting_mu,
                        starting_var=self.starting_var,
                        noise_var=self.noise_var,
                        halluc_sample=self.sample_mean() if hallucinate_mean else self.sample())

    def prob_func(self):
        # Return probability of a sample
        return partial(h_prob_func, 
                        running_total=self.running_total,
                        counts=self.counts, 
                        num_pops=self.num_pops, 
                        hallucinate_samples=0,
                        starting_mu=self.starting_mu,
                        starting_var=self.starting_var,
                        noise_var=self.noise_var,
                        halluc_sample=None)

    def sample_prob_func(self):
        # Return probability of a sample
        return partial(h_sample_prob_func, 
                        running_total=self.running_total,
                        counts=self.counts, 
                        num_pops=self.num_pops, 
                        hallucinate_samples=0,
                        starting_mu=self.starting_mu,
                        starting_var=self.starting_var,
                        noise_var=self.noise_var,
                        halluc_sample=None)

    def stats(self):
        # For debugging and logging return some statistics about payoff matrix
        # Return mean and variance across all entires of the payoff matrix
        return np.copy(self.means), np.copy(self.var)


def _build_mean_var(strat, running_total, counts, num_pops, hallucinate_samples, starting_mu, starting_var, noise_var, halluc_sample):
    new_counts = counts + 0
    new_running_total = running_total + 0
    if halluc_sample is not None:
        for p in range(num_pops):
            new_counts[p][strat] += hallucinate_samples
            new_running_total[p][strat] += halluc_sample[p][strat] * hallucinate_samples

    new_means = (noise_var * starting_mu + starting_var * new_running_total)\
                / (noise_var + new_counts * starting_var)
    new_vars = (noise_var * starting_var)\
                / (noise_var + new_counts * starting_var)
    
    return new_means, new_vars


# --- These functions are outside of the class in order to make them useable by a multiprocessing pool ---

def _norm_log_pdf(means, var, x, pops):
    pops = max(pops, 2) # 1 population matrices have the same number of dimensions as 2 population. e.g. (1,4,4) and (2,4,4). 
    stds = var**(1/2)
    denom = stds*(2*np.pi)**.5
    num = np.exp(-(x-means)**2/(2*var))
    return np.log((num/denom.clip(min=1e-100)).clip(min=1e-100)).sum(axis=tuple(range(-1, -(pops+2),-1)))


def h_sample_func(strat, running_total, counts, num_pops, hallucinate_samples, starting_mu, starting_var, noise_var, halluc_sample):
    new_means, new_vars = _build_mean_var(strat, running_total, counts, num_pops, hallucinate_samples, starting_mu, starting_var, noise_var, halluc_sample)
    m_sampled = np.random.normal(new_means, np.sqrt(new_vars))
    return m_sampled


def h_sample_prob_func(strat, running_total, counts, num_pops, hallucinate_samples, starting_mu, starting_var, noise_var, halluc_sample):
    new_means, new_vars = _build_mean_var(strat, running_total, counts, num_pops, hallucinate_samples, starting_mu, starting_var, noise_var, halluc_sample)
    m_sampled = np.random.normal(new_means, np.sqrt(new_vars))
    m_log_prob = _norm_log_pdf(new_means, new_vars, m_sampled, num_pops)
    return m_sampled, m_log_prob


def h_prob_func(strat, m_sample, running_total, counts, num_pops, hallucinate_samples, starting_mu, starting_var, noise_var, halluc_sample):
    new_means, new_vars = _build_mean_var(strat, running_total, counts, num_pops, hallucinate_samples, starting_mu, starting_var, noise_var, halluc_sample)
    m_log_prob = _norm_log_pdf(new_means, new_vars, m_sample, num_pops)
    return m_log_prob

