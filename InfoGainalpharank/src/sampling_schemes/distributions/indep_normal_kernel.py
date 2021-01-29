import numpy as np
import logging
from functools import partial


"""
This class implements the Gaussian Process used in the experiments in Figure 8 when providing prior knowledge.
The prior and GP are hard-coded for the 3 Good, 5 Bad matrix game when encoding the prior knowledge that:
1) M(s,t) + M(t,s) = 1
2) Entries in their respective blocks are equal to each other (except the top-left block, where they are just similiar)
"""
class NormalKernel:

    def __init__(self, num_pops, num_strats, num_players, starting_mu=0.5, starting_var=1, hallucination_samples=1, noise_var=1, estimate_noise=False, kernel=None):
        self.num_strats = num_strats
        self.num_players = num_players
        self.num_pops = num_pops # num_pops != num_players always. For single population we have num_pops=1 and num_players=2 (or more in general)
        assert self.num_pops == 1

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

        self.logger = logging.getLogger("Kernel")
        self.logger_count = 0

        # Hard coded kernel for 3 good, 5 bad
        if kernel is None:
            self.kernel = np.zeros(shape=(self.num_strats**2, self.num_strats**2))
            for i in range(self.num_strats):
                for j in range(self.num_strats):
                    for ix in range(self.num_strats):
                        for jx in range(self.num_strats): 

                            val = -1
                            if i < 3 and j < 3:
                                val = 0
                            if 3 <= i and j < 3:
                                val = 1
                            if 3 <= j and i < 3:
                                val = 2
                            if 3 <= i and 3 <= j:
                                val = 3

                            valx = -6
                            if ix < 3 and jx < 3:
                                valx = 0
                            if 3 <= ix and jx < 3:
                                valx = 1
                            if 3 <= jx and ix < 3:
                                valx = 2
                            if 3 <= ix and 3 <= jx:
                                valx = 3

                            kernel_val = 0 if val == valx else 1

                            index_1 = self.num_strats * i + j
                            index_2 = self.num_strats * ix + jx

                            if val == 0 and valx == 0:
                                kernel_val = 0

                            kernel_val = 0 if val == valx else 1
                            if val == 0 and valx == 0:
                                kernel_val = 1
                            if index_1 == index_2:
                                kernel_val = 0
                            self.kernel[index_1, index_2] = kernel_val

            # Encoding anti-symmetry
            self.new_kernel = self.kernel + 0
            for i in range(self.num_strats):
                for j in range(self.num_strats):
                    for ix in range(self.num_strats):
                        for jx in range(self.num_strats):
                            index_1 = self.num_strats * i + j
                            i1_r = self.num_strats * j + i
                            index_2 = self.num_strats * ix + jx
                            i2_r = self.num_strats * jx + ix
                            self.new_kernel[index_1, index_2] = self.kernel[index_1, index_2] - self.kernel[index_1, i2_r] - self.kernel[i1_r, index_2] + self.kernel[i1_r, i2_r]
            self.kernel = self.new_kernel

            self.kernel = self.kernel.dot(self.kernel.T)/500
        else:
            self.kernel = kernel

        self.noise_var = noise_var
        self.mu_start = starting_mu

        self.K = None
        self.K_inv = None
        self.f = None

        self.data = []

    def update_entry(self, strats, payoffs):
        strat_index = self.means.shape[1] * strats[0] + strats[1]
        self.data.append( (strat_index, payoffs[0]) ) # 1 population for simplicity

        if self.K is None:
            self.K = np.zeros(shape=(1,1)) + self.kernel[strat_index, strat_index] + self.noise_var
        else:
            N = len(self.data)-1
            new_k = np.zeros(shape=(N+1,N+1))
            new_k[:N,:N] = self.K
            for i in range(N):
                i_strat_index = self.data[i][0] 
                k_i_n = self.kernel[i_strat_index, strat_index]
                new_k[i, N] = k_i_n
                new_k[N, i] = k_i_n
            new_k[N, N] = self.kernel[strat_index, strat_index] + self.noise_var
            self.K = new_k

        self.K_inv = None # Since we have updated K, the inverse already computed is invalid
        self.f = None

    def info_gain_entry(self):
        new_mu, new_cov = self._calc_mu_cov()
        max_index = max([(new_cov[j,j],j) for j in range(new_cov.shape[0])])[1]
        s2 = max_index % self.num_strats
        s1 = (max_index-s2)//self.num_strats
        return (s1, s2)

    def _calc_mu_cov(self):
        if self.K is None:
            sampling_vector = np.array([i for i in range(self.num_strats**2)])

            k_ss = np.zeros(shape=(sampling_vector.shape[0], sampling_vector.shape[0]))
            for i,x in enumerate(sampling_vector):
                for j,y in enumerate(sampling_vector):
                    k_ss[i,j] = self.kernel[x,y]

            mu_s = np.zeros_like(sampling_vector) + self.mu_start

            new_mu = mu_s
            new_cov = k_ss

            return new_mu, new_cov
        else:
            if self.K_inv is None:
                self.K_inv = np.linalg.inv(self.K)
            if self.f is None:
                self.f = np.array([x[1] for x in self.data])

            sampling_vector = np.array([i for i in range(self.num_strats**2)])

            k_s_t = np.zeros(shape=(sampling_vector.shape[0], self.K.shape[1]))
            for i,x in enumerate(sampling_vector):
                for j,y in enumerate(self.data):
                    k_x_y = self.kernel[x, y[0]]
                    k_s_t[i,j] = k_x_y

            k_ss = np.zeros(shape=(sampling_vector.shape[0], sampling_vector.shape[0]))
            for i,x in enumerate(sampling_vector):
                for j,y in enumerate(sampling_vector):
                    k_ss[i,j] = self.kernel[x,y]

            mu_s = np.zeros_like(sampling_vector) + self.mu_start

            new_mu = mu_s + k_s_t.dot(self.K_inv).dot(self.f - self.mu_start)#[:,0]
            new_cov = k_ss - k_s_t.dot(self.K_inv).dot(k_s_t.T)

            return new_mu, new_cov

    def sample(self):
        new_mu, new_cov = self._calc_mu_cov()
        sampled_entries = np.random.multivariate_normal(new_mu, new_cov).reshape(1, self.num_strats, self.num_strats)
        return sampled_entries

    def prob(self, m_sample):
        raise NotImplementedError
    
    def sample_prob(self):
        raise NotImplementedError
    
    def sample_mean(self):
        new_mu, new_cov = self._calc_mu_cov()
        return np.copy(new_mu.reshape(1, self.num_strats, self.num_strats))

    def hallucinate_sample_func(self, hallucinate_mean=True):
        # Hallucinate a sample (either the mean or a sampled element) and return the resulting payoff matrix
        mus = {}
        covs = {}
        halluc_sample = self.sample_mean() if hallucinate_mean else self.sample() + np.random.normal(np.zeros_like(self.means), self.noise_var)
        for s in [(i,j) for i in range(self.num_strats) for j in range(self.num_strats)]:
            new_mu, new_cov = _build_mean_var(s, self.hallucination_samples, self.starting_mu, self.noise_var, halluc_sample, self.data, self.num_strats, self.kernel)
            mus[s] = new_mu
            covs[s] = new_cov
        return partial(h_sample_func, mus=mus, covs=covs, actions=self.num_strats)

    def stats(self):
        # For debugging and logging return some statistics about payoff matrix
        # Return mean and variance across all entires of the payoff matrix
        return np.copy(self.means), np.copy(self.var)


def _build_mean_var(strat, hallucinate_samples, starting_mu, noise_var, halluc_sample, data, actions, kernel):
    strat_index = actions * strat[0] + strat[1]
    new_data = data + [(strat_index, halluc_sample[0][strat]) for _ in range(hallucinate_samples)]
    N = len(new_data)

    K = np.zeros(shape=(N, N))
    for i, x in enumerate(new_data):
        for j, y in enumerate(new_data):
            K[i,j] = kernel[x[0], y[0]]
    for i in range(N):
        K[i,i] += noise_var

    K_inv = np.linalg.inv(K)

    f = np.array([x[1] for x in new_data])

    sampling_vector = np.array([i for i in range(actions**2)])

    k_s_t = np.zeros(shape=(sampling_vector.shape[0], K.shape[1]))
    for i,x in enumerate(sampling_vector):
        for j,y in enumerate(new_data):
            k_x_y = kernel[x, y[0]]
            k_s_t[i,j] = k_x_y

    k_ss = np.zeros(shape=(sampling_vector.shape[0], sampling_vector.shape[0]))
    for i,x in enumerate(sampling_vector):
        for j,y in enumerate(sampling_vector):
            k_ss[i,j] = kernel[x,y]

    mu_s = np.zeros_like(sampling_vector) + starting_mu

    new_mu = mu_s + k_s_t.dot(K_inv).dot(f - starting_mu)#[:,0]
    new_cov = k_ss - k_s_t.dot(K_inv).dot(k_s_t.T)

    return new_mu, new_cov

def h_sample_func(strat, mus, covs, actions):
    new_mu = mus[strat]
    new_cov = covs[strat]
    sampled_entries = np.random.multivariate_normal(new_mu, new_cov).reshape(1, actions, actions)
    return sampled_entries
