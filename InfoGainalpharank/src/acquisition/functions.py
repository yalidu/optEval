import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from scipy.spatial import Voronoi, ConvexHull
from scipy.stats import rankdata
#import ndd

# Acquisition functions, only 2 entropy variants are used from here. The wasserstein with L1 cost is implemented in wasserstein/utils.py

def l1(x,y):
    return np.abs(x-y).sum()

def jsd(x,y):
    return jensenshannon(x,y)**2 # Scipy returns the square root

def kendall_partial(p1, p2, bucket_decimals=2):
    # Computes the partial Kendall metric for p=0.5
    p = 1/2
    kendall_p = 0
    agents = len(p1)
    r1 = rankdata(np.around(p1,bucket_decimals), method="min")
    r2 = rankdata(np.around(p2,bucket_decimals), method="min")
    # Check r1 and r2 are ints
    for i in range(agents):
        for j in range(i+1, agents): # Values for i and j are the same, no need to compute them twice
            if i == j:
                continue
            if r1[i] == r1[j] and r2[i] != r2[j]:
                kendall_p += p
            elif r2[i] == r2[j] and not r1[i] != r1[j]:
                kendall_p += p
            else:
                # Appear in different orders
                if (r1[i] < r1[j] and r2[i] > r2[j]) or ( r1[i] > r1[j] and r2[i] < r2[j]):
                    kendall_p += 1
    return 2*kendall_p

def empirical_entropy(dist, bins=10):
    # Histogram of counts across the bins
    hist_counts, _ = np.histogramdd(dist, bins=bins, range=[(0,1) for _ in dist[0]])
    return entropy(hist_counts.flatten()) # Scipy will normalise 

def empirical_entropy_finite_support(dist, decimals=2):
    # Histogram of counts across the bins
    _, counts = np.unique(np.around(dist, decimals=decimals), return_counts=True, axis=0)
    return entropy(counts) # Scipy will normalise 
'''
def ndd_entropy(dist, k, decimals=100):
    _, counts = np.unique(np.around(dist, decimals=decimals), return_counts=True, axis=0)
    if counts.shape[0] == 1:
        # NDD doesn't accept a count vector over 1 thing
        counts = np.concatenate([counts, [0]], axis=0)
    return ndd.entropy(counts, k=k)
'''
def std_dev(dist):
    # Estimated standard deviation of population based on these samples
    mu = np.mean(dist, axis=0)
    sample_var = ((dist - mu) * (dist - mu)).sum()/(dist.shape[0]-1)
    return np.sqrt(sample_var)

def voronoi_log_volumes(p):
    # Too slow for large(ish) dimensions
    try:
        v = Voronoi(p + np.random.normal(0, 0.001, size=p.shape))
    except:
        return 0
    n = p.shape[0]
    denom = 0
    log_volume = 0
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 not in indices: # some regions can be opened
            log_volume += np.log(n*ConvexHull(v.vertices[indices]).volume)
            denom += 1
    return log_volume / denom