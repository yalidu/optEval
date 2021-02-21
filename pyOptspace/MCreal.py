import sys
#import os
sys.path.append("..")
print(sys.path)
import optspace
import time
from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import product
import numpy as np
import numpy.random as npr
import random
import pickle
import math
from random import shuffle
from alpha_rank import alpha_rank
import matplotlib.pyplot as plt
from functools import partial


from alpharank1 import iconstruct,compute
import scipy.stats
U = npr.randn(20, 4)
V = npr.randn(4, 20)
mat = np.dot(U, V)
plt.title('errors with rank')


plt.figure(13,figsize=(12,4))

plt.subplot(131)
plt.xlabel("samples m")
plt.ylabel("M_error")
plt.subplot(132)
plt.xlabel("samples m")
plt.ylabel("pi_error")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

plt.subplot(133)
plt.xlabel("samples m")
plt.ylabel("alpha_rank_error")
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
def PBS(M,pi):
    n=M.shape[-1]
    M=np.reshape(M,(n,n))
    pbs=np.zeros(n)
    for i in range(n):
        sump=0
        for j in range(n):
            if(M[i][j]>M[j][i]):
                sump+=pi[j]
        pbs[i]=sump
    return pbs
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

alpha=0.001
alpha_rank_partial=partial(alpha_rank,
                                 alpha=alpha,
                                 mutation=50,
                                 use_inf_alpha=False,
                                 inf_alpha_eps=0.0001,
                                 use_sparse=False,
                                 use_cache=True)

'''
d="/home/yali/yanxue/AlphaStarPayoffs.pkl"
with open(d, "rb") as f:
    mat = pickle.load(f)
#mat=np.load('/home/yali/yanxue/soccer200.npy')
#mat=np.load('/home/yali/yanxue/soccer200.npy')
#mat=np.log(mat/(1.0-mat))

C=iconstruct(mat)
print(np.linalg.matrix_rank(mat))

n=mat.shape[0]
'''
def MC(r,m,mat):
                true_alpha_rank = alpha_rank_partial(np.array([mat]))
                n=mat.shape[0]
                start_time=time.time()
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
                print(len(smat))
                (X, S, Y) = optspace.optspace(smat, rank_n=r,
                                              num_iter=10000,
                                              tol=1e-4,
                                              verbosity=1,
                                              outfile=b""
                                              )

                [X, S, Y] = map(np.matrix, [X, S, Y])
                bM = X * S * Y.T

                mmse = np.sqrt(np.sum(np.power(bM - mat, 2)) / X.shape[0] / Y.shape[0])
                pi_r = alpha_rank_partial(np.array([bM]))
                pi_max = np.max(np.abs(pi_r - true_alpha_rank))
                rank_error = calrank(true_alpha_rank, pi_r,1e-10)
                rank_error10=calrank(true_alpha_rank,pi_r,1e-15)
                pbst = np.max(PBS(mat, true_alpha_rank))
                pbse = np.max(PBS(mat, pi_r))
                mse_pi_error = np.abs(pbst - pbse)  #
                end_time=time.time()
                print("MC algorithm time:",end_time-start_time)
                return [mmse,pi_max,rank_error,rank_error10,mse_pi_error],end_time-start_time

def picture(m,X,label):
    print(X)
    print(X.shape)
    rep=X.shape[0]
    Merror=X[:,0,:].reshape(rep,-1)
    pierror=X[:,1,:].reshape(rep,-1)
    rankerror=X[:,2,:].reshape(rep,-1)
    #msepierror=X[:,2,:].reshape(3,-1)
    X=np.array(m)
    se = scipy.stats.sem(Merror, axis=0)
    plt.subplot(131)

    plt.fill_between(X, Merror.mean(0) - se, Merror.mean(0) + se,
                     zorder=10, alpha=0.2)
    plt.plot(X, Merror.mean(0), label=label, zorder=10)
    # plt.plot(X[0], X[1],label=label)

    # ax1.set_title("M error")
    plt.subplot(132)

    se = scipy.stats.sem(pierror, axis=0)
    plt.fill_between(X, pierror.mean(0) - se, pierror.mean(0) + se,
                     alpha=0.2)
    plt.plot(X, pierror.mean(0), label=label)
    # ax2.set_title("C error")
    plt.subplot(133)

    se = scipy.stats.sem(rankerror, axis=0)
    plt.fill_between(X, rankerror.mean(0) - se, rankerror.mean(0) + se,
                     alpha=0.2)
    plt.plot(X, rankerror.mean(0), label=label)


    #
def run_exp(d):
    i, exp,mat = d
    exp_data={}
    exp_data["MCr"]=exp["MCr"]
    exp_data["MCm"]=exp["MCm"]
    exp_data["rep"]=exp["rep"]
    exp_data["name"] = exp["name"]
    exp_data["data"],exp_data["times"]=MC(exp_data["MCr"],exp_data["MCm"],mat)#[1,2,3]
    return exp_data
def rungame(mat,r,name):
    n=mat.shape[0]
    rlist=[r]
    for i in range(n//20,n,n//5):
        if len(rlist)<5:
            rlist.append(i)
    mlist=[]
    for i in range(n*2,n*n,n*n//10):
        mlist.append(i)
    rlist = [30,35,40,45,50]
    mlist=[66*10,66*20,66*30,66*40,66*50,66*60]
    rlist = [ 12, 15,18,21]
    mlist = [21 * 4, 21 * 8, 21 * 12, 21 * 14,21 * 16, 21*18, 21*20,21*21-10,21*21-1]
    #rlist = [10,14, 20, 25, 30]
    #mlist = [n*i for i in range(20,160,20)]#[21 * 4, 21 * 8, 21 * 12, 21 * 14, 21 * 16, 21 * 18, 21 * 20, 21 * 21]
    #rlist = [2, 4, 6, 8]
    #mlist = [n * i for i in range(1,15, 2)]
    #rlist = [38, 45, 50, 60]
    #mlist = [n * i for i in range(20, 160, 20)]
    with open("generm.pkl", "rb") as f:
        rm=pickle.load( f)
    #rlist = [8, 18, 30,40,50]
    #mlist = [n * i for i in range(20, 889, 80)]
    #rlist = [12, 15, 18, 21]
    #mlist = [21 * 6, 21 * 8, 21 * 12,21*13, 21 * 14, 21 * 16, 21 * 18, 21 * 20]
    rlist=rm[sys.argv[1]][0]
    mlist=rm[sys.argv[1]][1]
    base_params = {
        # Alpha rank hyper-parameters
        "name":[name],
        "MC": [True],
        "MCr": rlist,
        #"MCm":[5000]+[ i for i in range(10000,100001,10000)],
        "MCm": mlist,
    }
    exp_params = [{"rep":[1,2]}]
    exps=[]

    for exp_dict in exp_params:
        full_dict = {**base_params,**exp_dict}

        some_exps = list(map(dict, product(*[[(k, v) for v in vv] for k, vv in full_dict.items()])))
        exps = exps + some_exps
    print(len(exps))
    print(exps[0])
    num_in_parallel =32#int(sys.argv[2])
    start_time = time.time()
    print("--- Starting {} Experiments ---".format(len(exps)))
    r_line={}
    with Pool(num_in_parallel) as pool:
        ix = 0
        for exp_data in pool.map(run_exp, [(i, exp,mat) for i, exp in enumerate(exps)]):
            ix += 1
            print(exp_data)
            r=exp_data["MCr"]
            if r not in r_line.keys():
                r_line[r]={}
            m=exp_data["MCm"]
            if m not in r_line[r].keys():
                r_line[r][m]=[]
            data=exp_data["data"]
            r_line[r][m].append(data)
    end_time=time.time()
    print("end experiments in",end_time-start_time)
    with open("../picture_data/{}/18gamedata/redraw/{}.pkl".format(alpha,sys.argv[1]), "wb") as f:
        pickle.dump(r_line, f)
    for r in base_params["MCr"]:
        omega=[]
        for m in base_params["MCm"]:
            omega.append(r_line[r][m])
        omega=np.array(omega)
        omega=omega.transpose((1,2,0))
        picture(base_params["MCm"],omega,label="r={}".format(r))
    #X=MC()
    #picture(X)
    plt.subplot(131)
    plt.grid()
    plt.subplot(132)
    plt.grid()
    plt.subplot(133)
    plt.grid()
    plt.legend()

    plt.savefig('../picture_data/{}/18gamedata/redraw/12game/{}.pdf'.format(alpha,sys.argv[1]))
    plt.show()
if __name__=='__main__':
    with open("/data/yanxue/spinning_top_payoffs.pkl", "rb") as fh:
        payoffs = pickle.load(fh)

    #sys.argv.append('AlphaStar1')
    game_name = sys.argv[1]
    mat=payoffs[game_name]

    rank = np.linalg.matrix_rank(mat)
    rungame(mat,rank,name=game_name)
    '''
    n=888
    with open("../picture_data/18gamedata/AlphaStar1.pkl", "rb") as f:
        r_line=pickle.load(f)

    rlist = [8, 18, 30, 40, 50]
    mlist = [n * i for i in range(20, 889, 80)]
    for r in rlist:
        omega=[]
        for m in mlist:
            omega.append(r_line[r][m])
        omega=np.array(omega)
        omega=omega.transpose((1,2,0))
        picture(mlist,omega,label="r={}".format(r))
    #X=MC()
    #picture(X)
    plt.subplot(131)
    plt.grid()
    plt.subplot(132)
    plt.grid()
    plt.subplot(133)
    plt.grid()
    plt.legend()

    plt.savefig('../picture_data/18gamep/{}.pdf'.format(sys.argv[1]))
    plt.show()
    '''
