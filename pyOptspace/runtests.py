import optspace
import numpy as np
import numpy.random as npr
import sys

sys.path.append("..")

import random
import math
import pickle
from random import shuffle
#from .. import alpha_rank
from sampling import calrank
import matplotlib.pyplot as plt
from functools import partial
from alpha_rank import alpha_rank

alpha_rank_partial=partial(alpha_rank,
                                 alpha=100,
                                 mutation=50,
                                 use_inf_alpha=False,
                                 inf_alpha_eps=0.0001,
                                 use_sparse=False,
                                 use_cache=True)
U = npr.randn(200, 10)
V = npr.randn(10, 200)
mat = np.dot(U, V)
mat=np.zeros((200,200))

with open("../NeurIPS/spinning_top_payoffs.pkl", "rb") as fh:
    payoffs = pickle.load(fh)

sys.argv.append('AlphaStar')
game_name = sys.argv[1]
mat = payoffs[game_name].reshape(1,888,888)
print(mat[0][821][881])
pi=alpha_rank_partial(mat)
print(np.argsort(-pi))
pi=-np.sort(-pi)
print(pi[:20])
smat=[]
for i in range(21):
    for j in range(21):
       #if random.randint(1,1)<=1:
            smat.append((i,j,mat[i][j]))
(X, S, Y) = optspace.optspace(smat, rank_n=12,
                                              num_iter=10000,
                                              tol=1e-4,
                                              verbosity=1,
                                              outfile=b""
                                              )

[X, S, Y] = map(np.matrix, [X, S, Y])
bM = X * S * Y.T
print(np.sum(np.square(mat-bM)))
#821 881 885 832 851 792 861 801 870 853 842 873 781 752 886 844 884 834
#771 681 812 864 631 760 741 710 805 785 837 823 556 731 722 833 648 846



