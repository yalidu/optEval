import numpy as np
import logging


def run_sampling(payoff_matrix_sampler, sampler, max_iters=100, graph_samples=10, true_alpha_rank=None, true_payoff=None):
    logger = logging.getLogger("Sampling")
    logger.warning("Starting sampling for up to {:,} iterations.".format(max_iters))
    
    # Quantities to log
    improvements = []
    entries = []
    alpha_ranking_distrib = []
    alpha_ranking_distrib_more = []
    mean_alpha_ranks = []
    prob_alpha_ranks = []
    prob_alpha_rank_ps = []
    payoff_matrix_means = []
    payoff_matrix_vars = []

    for t in range(max_iters):
        # Log current (approximation) of distribution over alpha-rankings
        if t % (max_iters//10) == 0:
            print(t)

        # Pick an entry to sample
        entry_to_sample, sampler_stats = sampler.choose_entry_to_sample()
        if entry_to_sample is None:
            logger.info("Finished sampling at {} iterations".format(t))
            break
        #entries.append(entry_to_sample)
        #if "improvements" in sampler_stats:
        #    improvements.append(sampler_stats["improvements"])
        #logger.info("Sampling {}".format(entry_to_sample))
        
        # Get a sample from that entry
        payoff_samples = payoff_matrix_sampler.get_entry_sample(entry_to_sample)
        #print(entry_to_sample,payoff_samples)

        #logger.info("Received Payoff {} for {}".format(payoff_samples, entry_to_sample))

        # Update entry distribution with this new sample
        sampler.update_entry(entry_to_sample, payoff_samples)
    last_means, m_vars = sampler.payoff_distrib()
    print(last_means)
    print(np.sum(np.square(last_means-payoff_matrix_sampler.true_payoffs())))
    logger.critical("Finished {} iterations".format(max_iters))
    del sampler

    return {
        '''
        "improvements": improvements,
        "entries": entries,
        "mean_alpha_rankings": mean_alpha_ranks,
        "prob_alpha_rankings": prob_alpha_ranks,
        "prob_alpha_rank_ps": prob_alpha_rank_ps,
        "alpha_rankings": alpha_ranking_distrib,
        "alpha_rankings_more": alpha_ranking_distrib_more,
        "payoff_matrix_means": payoff_matrix_means,
        "payoff_matrix_vars": payoff_matrix_vars,
        '''
        "last_means":last_means
    }
