import sys
sys.path.append("/home/yali/yanxue/pyOptspace")
print(sys.path)
import numpy as np
import logging
import optspace
import random

def Optspace(mat,r,mask):
    n=mat.shape[1]
    mat=np.reshape(mat,(n,n))
    smat = []
    pairs = mask
    for pair in pairs:
        i = pair[0]
        j = pair[1]
        smat.append((i, j, mat[i][j]))
        #smat.append((j, i, -mat[i][j]))
    #for i in range(n):
    #    smat.append((i, i, 0))
    (X, S, Y) = optspace.optspace(smat, rank_n=r,
                                  num_iter=50000,
                                  tol=1e-4,
                                  verbosity=1,
                                  outfile="")

    [X, S, Y] = map(np.matrix, [X, S, Y])
    bC = np.array(X * S * Y.T)
    bC =np.reshape(bC,(1,n,n))
    return bC
eps=1e-20
def calrank(pi,pi_hat,eps=1e-20):
    #print(pi,pi_hat)
    n=pi.shape[0]
    #eps=1/n/n

    rerror=0
    for i in range(n):
        for j in range(n):
            if i !=j :#and pi[i]>eps and pi[j]>eps:
                if pi[i]-pi[j]>eps:
                    if pi_hat[i]-pi_hat[j]<-eps:
                        rerror+=1
                else:
                    if pi[i] -pi[j] < -eps:
                        if pi_hat[i]-pi_hat[j]>eps:
                            rerror+=1
    return 1.0*rerror/n/n

def run_sampling(payoff_matrix_sampler, sampler, max_iters=100, graph_samples=10,P1=0, true_alpha_rank=None,
                 true_payoff=None,mask=None,r=None,alpha_rank_func=None):
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
    payoff_matrix_error=[]
    alphi_error=[]
    alpha_rank_error=[]
    iterlist=[]
    global last_payoff
    global last_mean_alpha_rank
    t=0
    #max_iters=5
    #while True:
    for t in range(max_iters+1):
        # Log current (approximation) of distribution over alpha-rankings
        t+=1
        if t%100000==0:
            print(t)




        # Pick an entry to sample
        entry_to_sample, sampler_stats = sampler.choose_entry_to_sample()
        if entry_to_sample is None:
            print("stop sampling at ",t)
            break
        '''
        if entry_to_sample is None:
            logger.info("Finished sampling at {} iterations".format(t))
            break
        entries.append(entry_to_sample)
        if "improvements" in sampler_stats:
            improvements.append(sampler_stats["improvements"])
        '''
        #logger.info("Sampling {}".format(entry_to_sample))

        # Get a sample from that entry

        for k in range(1):
            payoff_samples = payoff_matrix_sampler.get_entry_sample(entry_to_sample)
        #logger.info("Received Payoff {} for {}".format(payoff_samples, entry_to_sample))

        # Update entry distribution with this new sample
            sampler.update_entry(entry_to_sample, payoff_samples)

    #print(last_payoff)

    last_payoff,vars = sampler.payoff_distrib()
    if P1==1:#convert payoff matrix into an antisymmetric matrix
        last_payoff=2*last_payoff-1
    n=last_payoff.shape[-1]
    #for i in range(n):
    #    last_payoff[0][i][i]=true_payoff[0][i][i]
    if mask is not None:
        last_payoff=Optspace(last_payoff,r,mask)
        #last_payoff = Optspace(true_payoff, r, mask)
        print("MC",np.sum(np.square(last_payoff-true_payoff)))
    print(np.sum(np.square(last_payoff - true_payoff)))
    merror=np.mean(np.square(last_payoff - true_payoff))
    true_alpha_rank=alpha_rank_func(true_payoff)
    last_alpha_rank=alpha_rank_func(last_payoff)
    alphi_error=np.max(np.abs(true_alpha_rank-last_alpha_rank))
    logger.critical("Finished {} iterations".format(max_iters))
    del sampler

    return {
        "alpha_error":alphi_error,
        "merror":merror,
        "payoff_matrix_error":payoff_matrix_error,
        "alpha_rank_error":alpha_rank_error,
        #"last_mean_alpha":last_mean_alpha_rank,
        "last_payoff":last_payoff,
        "iter_list":iterlist,
        "timest":t
        #"improvements": improvements,
        #"entries": entries,
        #"mean_alpha_rankings": mean_alpha_ranks,
        #"prob_alpha_rankings": prob_alpha_ranks,
        #"prob_alpha_rank_ps": prob_alpha_rank_ps,
        #"alpha_rankings": alpha_ranking_distrib,
        #"alpha_rankings_more": alpha_ranking_distrib_more,
        #"payoff_matrix_means": payoff_matrix_means,
        #"payoff_matrix_vars": payoff_matrix_vars,
    }