# Calculates the alpha rank of a provided matrix game, with some additional caching to speedup repeated computation.
# Based on the implementation in OpenSpiel (https://github.com/deepmind/open_spiel)
import numpy as np
import scipy.linalg as la
from copy import copy
from itertools import product
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs

# Caching computation
strat_to_index = None
cache_strats = None
cache_players = None


def alpha_rank(M, alpha=1, mutation=50, use_inf_alpha=False, inf_alpha_eps=0.01, use_sparse=False, use_cache=False):
    """
    Computes the stationary distribution over strategies associated with alpha rank.
    
    Arguments:
        M {List of numpy arrays} -- The payoff matrices for each population of agents we are considering. A single payoff matrix results in the single population model being used.
    
    Keyword Arguments:
        alpha {float > 0} -- The alpha parameter for alpha rank. (default: {1})
        mutation {int > 1} -- Inverse of the mutation rate. 1 / mutation is used as the probability of switching to another strategy if they have (roughly) equal payoffs. (default: {50})
        use_inf_alpha {boolean} -- Whether to use an infinite alpha and perturb the transition matrix to guarantee irreducibility
        inf_alpha_eps {0 < float < 1} -- Perturbation to use on transitions if using infinite alpha
    
    Raises:
        ValueError: Found more than 1 eigenvector with an eigenvalue near 1. Indicates a problem calculating the stationary distribution for the  Markov Chain. 
    
    Returns:
        phi {numpy array} -- The stationary distribution of alpha rank over the set of joint-strategies. 
    """    

    num_strats = M[0].shape[0] # The number of pure strategies/actions for an agent
    num_players = len(M)

    C = np.zeros(shape=(num_strats**num_players, num_strats**num_players))

    eta = 1 / (num_players * (num_strats - 1))

    global strat_to_index
    # Incase we call alpha rank on a differently sized matrix game
    global cache_strats
    global cache_players
    if strat_to_index is None or not use_cache or (strat_to_index is not None and (cache_strats, cache_players) != (num_strats, num_players)):
        strat_to_index = np.zeros(shape=[num_strats for _ in range(num_players)], dtype=np.uint16)
        for base_strat in product(range(num_strats), repeat=num_players):
            base_strat_index = sum([num_strats**ix * x for ix, x in enumerate(reversed(base_strat))])
            strat_to_index[base_strat] = base_strat_index
        cache_strats = num_strats
        cache_players = num_players

    for base_strat in product(range(num_strats), repeat=num_players):
        base_strat_index = strat_to_index[base_strat]
        # Now consider strategies that deviate in a single players change
        for n in range(num_players):
            for strat_index in range(num_strats):
                if strat_index == base_strat[n]:
                    continue # We will fill in the probability of sticking with the base strategy afterwards to ensure its a valid probability distribution
                new_strat = list(base_strat)
                new_strat[n] = strat_index
                new_strat_tuple = tuple(new_strat)
                new_strat_index = strat_to_index[new_strat_tuple]

                fitness_diff = _get_fitness_diff(M, num_players, n, new_strat_tuple, base_strat)

                transition_prob = 0
                if use_inf_alpha:
                    if abs(fitness_diff) < 0.0001: # Using this for speed, np.isclose() was taking a significant chunk of time
                        transition_prob = 0.5 * eta # From OpenSpiel
                    elif fitness_diff > 0:
                        transition_prob = eta * (1 - inf_alpha_eps)
                    else:
                        transition_prob = eta * inf_alpha_eps
                else:
                    if abs(fitness_diff) < 0.0001:
                        transition_prob = eta / mutation
                    else:
                        # Checking for overflow
                        if -alpha * mutation * fitness_diff > 500:
                            transition_prob = 0
                        else:
                            transition_prob = eta * (1 - np.exp(-alpha * fitness_diff)) / (1 - np.exp(-alpha * mutation * fitness_diff))

                C[base_strat_index, new_strat_index] = transition_prob

    # Ensure C is a valid transition matrix by filling in the diagonals
    for base_strat_index in range(C.shape[0]):
        C[base_strat_index, base_strat_index] = 1 - C[base_strat_index, :].sum()

    # Approximate the stationary distribution of C
    if use_sparse:
        C_sparse = csr_matrix(C.T) # Transposing C is very important here
        eigenvals, eigenvectors = eigs(C_sparse, k=1, which="LM")
        mask = abs(eigenvals - 1.) < 1e-10
        left_eigenvecs = eigenvectors[:, mask]
        num_stationary_eigenvecs = np.shape(left_eigenvecs)[1]
        if num_stationary_eigenvecs != 1:
            if use_inf_alpha:
                raise ValueError("{} stationary distributions found using Sparse Solver. Using infinite alpha with pertub: {:.8f}, Mutation: {}".format(num_stationary_eigenvecs, inf_alpha_eps, mutation))
            else:
                raise ValueError("{} stationary distributions found using Sparse Solver. Alpha: {:.2f}, Mutation: {}".format(num_stationary_eigenvecs, alpha, mutation))
        left_eigenvecs *= 1. / sum(left_eigenvecs) # Ensuring it is a valid probability distribution
        phi = left_eigenvecs.real.flatten()
    else:
        # Use an eigenvector solver from scipy, taken from OpenSpiel alpha_rank code.
        eigenvals, left_eigenvecs, _ = la.eig(C, left=True, right=True) # TODO: Can we remove the right eigenvectors?
        mask = abs(eigenvals - 1.) < 1e-10
        left_eigenvecs = left_eigenvecs[:, mask]
        num_stationary_eigenvecs = np.shape(left_eigenvecs)[1]
        if num_stationary_eigenvecs != 1:
            if use_inf_alpha:
                raise ValueError("{} stationary distributions found. Using infinite alpha with pertub: {:.8f}, Mutation: {}".format(num_stationary_eigenvecs, inf_alpha_eps, mutation))
            else:
                raise ValueError("{} stationary distributions found. Alpha: {:.2f}, Mutation: {}".format(num_stationary_eigenvecs, alpha, mutation))
        left_eigenvecs *= 1. / sum(left_eigenvecs) # Ensuring it is a valid probability distribution
        phi = left_eigenvecs.real.flatten()

    # --- Approximating the stationary distribution using the power method ---
    # phi = None
    # for i in range(10000):
    #     C_new = np.linalg.matrix_power(C, 100)
    #     if np.all(np.isclose(C, C_new, atol=1e-3)):
    #         phi = C_new[0]
    #         break
    #     C = C_new
    # if phi is None:
    #     # C didn't converge to the stationary distribution
    #     raise Exception("Need more iterations for convergence to stationary distribution.")
    # phi = phi / phi.sum()
    # ---

    return phi

def _get_fitness_diff(payoffs, num_players, player_changing, new_strat, base_strat):
    """
    Simple function for returning the difference between the payoffs of 2 strategies. 
    There is a difference in the calculation for 1 or more populations.    

    Arguments:
        payoffs {list of numpy arrays} -- The payoff matrices
        num_players {int} -- The number of populations
        player_changing {int} -- The player who is changing their strategy, needed to decide what payoff matrix to use.
        new_strat {tuple of ints} -- The new strategy we are considering. The tuple of indices to use for indexing the payoff matrix.
        base_strat {tuple of ints} -- The base strategy we are considering.
    
    Returns:
        Difference in payoffs [float] -- The difference in payoffs between the base and new strategy, used to calculate the probability of transitioning from base to new_strat.
    """
    M = payoffs
    if num_players == 1:
        return M[0][new_strat, base_strat] - M[0][base_strat, new_strat]
    else:
        n = player_changing
        return M[n][new_strat] - M[n][base_strat]

