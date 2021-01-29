import numpy as np
from itertools import product
from functools import partial
from wasserstein.utils import wasserstein_l1
from acquisition.functions import empirical_entropy_finite_support
#from acquisition.functions import ndd_entropy
import logging
from multiprocessing import Pool


class BayesianBandit:

    def __init__(self,
                 num_pops,
                 num_strats,
                 num_players=2,
                 payoff_distrib=None,
                 alpha_rank_func=None,
                 mc_samples=10,
                 acquisition=None,
                 expected_hallucinate=False,
                 expected_samples=10,
                 use_parallel=True,
                 repeat_sampling=1):

        self.num_strats = num_strats
        self.num_players = num_players
        self.num_pops = num_pops  # num_pops != num_players always. For single population we have num_pops=1 and num_players=2 (or more in general)

        assert alpha_rank_func is not None
        self.alpha_rank = alpha_rank_func

        assert acquisition is not None
        self.acquisition = acquisition  # String determining the acquisition to use

        self.payoffs = payoff_distrib

        self.samples = mc_samples
        self.total_samples = mc_samples

        self.repeat_sampling = repeat_sampling
        self.next_sample = (None, 0)  # Entry to sample next and number of more times to sample it
        self.random_iters = 0

        self.expected_hallucinate = expected_hallucinate
        self.expected_samples = expected_samples
        if not self.expected_hallucinate and self.expected_samples > 1:
            raise Exception("Using more than 1 sample when hallucinating the mean.")

        self.logger = logging.getLogger("Bayesian_Bandit")

        self.use_parallel = use_parallel
        if use_parallel:
            self.pool = Pool(initializer=_init_worker, initargs=(
            self.num_strats, self.samples, self.total_samples, self.num_pops, self.alpha_rank, self.acquisition,
            #self.flip_improvement
            ))

        _init_worker(self.num_strats, self.samples, self.total_samples, self.num_pops, self.alpha_rank,
                     self.acquisition)#, self.flip_improvement)

    def __del__(self):
        try:
            if self.use_parallel:
                self.pool.terminate()
        except AttributeError:
            pass  # Happens

    def choose_entry_to_sample(self):
        if self.random_iters < self.repeat_sampling:
            strat_to_sample = self.payoffs.info_gain_entry()
            self.random_iters += 1
            return strat_to_sample, {}
        if self.next_sample[1] > 0:
            strat_to_sample = self.next_sample[0]
            self.next_sample = (strat_to_sample, self.next_sample[1] - 1)
            return strat_to_sample, {}

        logging = {}

        strats = list(product(range(self.num_strats), repeat=self.num_players))

        base_phis = None
        if self.acquisition in ["l1_relative"]:  # Wasserstein with L1 cost
            base_phis = []
            for _ in range(self.total_samples):
                m = self.payoffs.sample()
                phi = self.alpha_rank(m)
                base_phis.append(phi)
            base_phis = np.array(base_phis)
        all_imps_and_entries = None
        for i in range(self.expected_samples):
            strat_improvement = partial(_get_entry_improvement,
                                        payoffs=self.payoffs.hallucinate_sample_func(
                                            hallucinate_mean=not self.expected_hallucinate),
                                        base_phis_to_use=base_phis)
            improvements_and_entries = self.pool.map(strat_improvement, strats) if self.use_parallel else list(
                map(strat_improvement, strats))
            if all_imps_and_entries is None:
                all_imps_and_entries = improvements_and_entries
            else:
                all_imps_and_entries = [(a, b + d) for (a, b), (c, d) in
                                        zip(all_imps_and_entries, improvements_and_entries)]
        all_imps_and_entries = [(a, b / self.expected_samples) for (a, b) in all_imps_and_entries]
        improvements = [x[1] for x in all_imps_and_entries]
        max_strat, max_value = max(all_imps_and_entries, key=lambda x: x[1])

        logging["improvements"] = np.array(improvements)
        self.logger.debug("Max improvement: {:.3f}, Mean: {:.3f}, Min: {:.3f}".format(max_value, np.mean(improvements),
                                                                                      np.min(improvements)))

        self.next_sample = (max_strat, self.repeat_sampling - 1)

        return max_strat, logging

    def update_entry(self, strats, payoffs):
        self.payoffs.update_entry(strats, payoffs)

    def alpha_rankings_distrib(self, graph_samples=None, mean=False):
        if mean:
            mean_payoffs = self.payoffs.sample_mean()
            phi = self.alpha_rank(mean_payoffs)
            return np.array([phi])
        else:
            # For debugging and logging approximate the distribution over alpha-rankings
            samples = [self.payoffs.sample() for _ in range(self.samples if graph_samples is None else graph_samples)]
            phis = self.pool.map(_alpha_rank_sample, samples) if self.use_parallel else list(
                map(_alpha_rank_sample, samples))  # To keep the same code for both

            # Also return the most probable alpha rank and its estimated probability
            unique_phis, counts = np.unique(np.around(phis, decimals=5), return_counts=True, axis=0)
            argmax_entry = np.argmax(counts)
            p_phi = unique_phis[argmax_entry]
            p_phi_prob = counts[argmax_entry] / sum(counts)

            return np.array(list(phis)), (np.array([p_phi]), p_phi_prob)

    def payoff_distrib(self):
        return self.payoffs.stats()


def _init_worker(_num_strats, _alpha_samples, _phi_samples, _num_pops, _alpha_rank, _cost_func):
    global num_strats
    global samples
    global num_pops
    global alpha_rank
    global cost_func
    global base_phi_sampler
    global k_upper_bound

    num_strats = _num_strats
    samples = _alpha_samples
    num_pops = _num_pops
    alpha_rank = _alpha_rank
    cost_func = _cost_func
    base_phi_sampler = partial(np.random.dirichlet, alpha=np.ones(num_strats ** num_pops), size=(_phi_samples,))
    k_upper_bound = 2 ** ((num_strats ** num_pops) * num_pops * (num_strats - 1))


def _alpha_rank_sample(m_sample):
    phi = alpha_rank(m_sample)
    return phi


def _get_alpha_rank(strat, payoffs, samples_to_gather=None):
    phis = []
    m_samples = []
    m_probs = []
    for _ in range(samples if samples_to_gather is None else samples_to_gather):
        m_sampled, m_prob = payoffs(strat)
        phi = alpha_rank(m_sampled)
        phis.append(phi)
        m_samples.append(m_sampled)
        m_probs.append(m_prob)
    phis = np.array(phis)
    m_samples = np.array(m_samples)
    m_probs = np.array(m_probs)
    data_to_return = (phis, m_samples, m_probs)

    return data_to_return


def _get_improvement(phis, base_phis_to_use=None):
    # Calculate the improvement/acquisition cost
    improvement = None
    if cost_func == "l1_relative":
        base_phis = base_phis_to_use
        improvement = wasserstein_l1(base_phis, phis, normalise=False)
    elif cost_func == "entropy_support":
        improvement = -empirical_entropy_finite_support(phis)
    elif cost_func == "ndd_entropy":
        improvement = -empirical_entropy_finite_support(phis)
        #improvement = -ndd_entropy(phis, k=k_upper_bound, decimals=5)
    else:
        raise Exception("Incorrect acquisition function specified: {}!".format(cost_func))

    return improvement


def _get_entry_improvement(strat, payoffs, base_phis_to_use=None):
    phis = []
    for _ in range(samples):
        m_sampled = payoffs(strat)
        phi = alpha_rank(m_sampled)
        phis.append(phi)
    phis = np.array(phis)

    # Calculate the improvement/acquisition cost
    improvement = _get_improvement(phis, base_phis_to_use=base_phis_to_use)

    return strat, improvement