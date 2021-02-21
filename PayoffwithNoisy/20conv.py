import numpy as np
import time
import random
from random import shuffle
import sys
sys.path.append("/home/yali/yanxue/pyOptspace")
import optspace
from functools import partial
from alpha_rank import alpha_rank
import matplotlib.pyplot as plt

alpha=0.001
alpha_rank_partial=partial(alpha_rank,
                                 alpha=alpha,
                                 mutation=50,
                                 use_inf_alpha=False,
                                 inf_alpha_eps=0.0001,
                                 use_sparse=False,
                                 use_cache=True)
def MC(r, m,mat):
    true_alpha_rank=alpha_rank_partial([mat])
    start_time = time.time()
    smat = []
    pairs = []

    for i in range(mat.shape[0]):
        j = random.randint(0, mat.shape[1] - 1)
        pairs.append((i, j))
    for j in range(mat.shape[1]):
        i = random.randint(0, mat.shape[0] - 1)
        if (i, j) not in pairs:
            pairs.append((i, j))
    rest = m - len(pairs)

    select = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if (i, j) not in pairs:
                select.append((i, j))
    if rest > 0:
        shuffle(select)
        rest = min(rest, len(select))
        for i in range(rest):
            pairs.append(select[i])

    for pair in pairs:
        i = pair[0]
        j = pair[1]
        smat.append((i, j, mat[i][j]))

    (X, S, Y) = optspace.optspace(smat, rank_n=r,
                                  num_iter=50000,
                                  tol=1e-4,
                                  verbosity=1,
                                  outfile=b""
                                  )

    [X, S, Y] = map(np.matrix, [X, S, Y])
    bM = X * S * Y.T

    mmse = np.sqrt(np.sum(np.power(bM - mat, 2)) / X.shape[0] / Y.shape[0])
    pi_r = alpha_rank_partial(np.array([bM]))
    pi_max = np.max(np.abs(pi_r - true_alpha_rank))
    #rank_error = calrank(true_alpha_rank, pi_r, eps=1e-20)
    #rank_error_10 = calrank(true_alpha_rank, pi_r, eps=1e-10)
    #pbst = np.max(PBS(mat, true_alpha_rank))
    #pbse = np.max(PBS(mat, pi_r))
    #mse_pi_error = pbst - pbse
    end_time = time.time()
    print("MC algorithm time:", end_time - start_time)
    return mmse, pi_max#, rank_error, rank_error_10, mse_pi_error], end_time - start_time


if __name__=='__main__':
    '''
    eps=1e-4
    for i in range(10,201,10):
        last=1.0
        X=[]
        Y=[]
        mat=np.load("conv{}.npy".format(i))
        for j in range(i,i*i+1,10):
            X.append(j)
            merror,pimax=MC(5,j,mat)
            Y.append(pimax)
            print(merror,pimax)
            if pimax<eps/i:
                print("converage at:",j)
                np.save("convm{}.npy".format(i),j)
                break
        plt.plot(X,Y)
    plt.show()
    '''
    X=[]
    Y=[]
    for i in range(10,201,10):
        j=np.load("convm{}.npy".format(i))
        X.append(i)
        Y.append(j)
    X=np.array(X)
    plt.xlabel('players $n$',fontsize=14)
    plt.ylabel('samples $m$',fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.scatter(X,Y,marker='+',label='optEval-1',s=40,color='violet')
    #plt.plot(X, X * 5 * np.log(X), label='nrlogn')
    #plt.plot(X, 0.8 * X * 5 * np.log(X), label='0.8nrlogn')

    plt.plot(X,0.6*X*5*np.log(X),label='0.6nrlogn')

    #
    plt.plot(X,0.5*X*5*np.log(X),label='0.5nrlogn')

    plt.plot(X, 0.4 * X * 5 * np.log(X), label='0.4nrlogn')
    plt.savefig("nrlogn2.pdf")

    plt.show()
    '''
    Y=[]
    X=[]
    for i in range(10, 201, 10):
        #np.save('./convnoisy/optm{}.npy'.format(players), convat)
        m=np.load('./convnoisy/optm{}.npy'.format(i))
        print(m)
        X.append(i)
        Y.append(m)
    #plt.scatter(X, Y, marker='*', label='optEval-2', s=40, color='gold')


    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig("nrlogn2.pdf")

    plt.show()
    '''

