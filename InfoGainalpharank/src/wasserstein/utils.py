from scipy.spatial.distance import jensenshannon
from ot import emd2
import numpy as np


def _l1_cost(x, y):
    # L1 as default
    return np.abs(x - y).sum()


def wasserstein_l1(base_dist, new_dist, normalise=False):
    """ Calculate the wasserstein distance between 2 distributions via samples using l1 cost"""

    # Calculate the Wasserstein distance between base_dist and new_dist using JSD as the cost
    N = base_dist.shape[0]

    base_dist = np.copy(base_dist)
    new_dist = np.copy(new_dist)
    
    # Sanitise the distributions
    base_dist.clip(0, 1)
    new_dist.clip(0, 1)

    base_vector = np.ones(N)/N
    new_vector = np.ones(N)/N
    cost_matrix = np.zeros(shape=(N,N))

    for i in range(N):
        row_cost = np.abs(base_dist[i] - new_dist).sum(axis=1)
        cost_matrix[i] = row_cost

    if normalise:
        mean_phi = np.mean(new_dist, axis=0)
        cost_to_base = np.abs(mean_phi - base_dist).sum(axis=1) + 0.0000001
        cost_matrix = cost_matrix / cost_to_base[:,np.newaxis]

    w_results = emd2(base_vector, new_vector, cost_matrix)

    return w_results
