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
from sampling import calrank
import matplotlib.pyplot as plt
from functools import partial
#from MCpara import PBS
from alpharank1 import iconstruct,compute
import scipy.stats
U = npr.randn(20, 4)
V = npr.randn(4, 20)
mat = np.dot(U, V)
plt.title('errors with rank')
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

fz=14
_, axes=plt.subplots(14,figsize=(8,6))

plt.subplot(221)

plt.xlabel("samples $m$",fontsize=fz)
plt.ylabel("$M$ error",fontsize=fz)
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

plt.subplot(222)
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
plt.xlabel("samples $m$",fontsize=fz)
plt.ylabel("$\pi$ error",fontsize=fz)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

plt.subplot(223)
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
plt.xlabel("samples $m$",fontsize=fz)
plt.ylabel(r"$\alpha$ rank ranking error",fontsize=fz)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

plt.subplot(224)
plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
plt.xlabel("samples $m$",fontsize=fz)
plt.ylabel(r"$\alpha$-Conv",fontsize=fz)
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

mat=np.load('../soccer200.npy')
#mat=np.load('/home/yali/yanxue/soccer200.npy')
#mat=np.log(mat/(1.0-mat))


#print(np.linalg.matrix_rank(mat))

#Mmax=np.max(mat)
#Mmin=np.min(mat)
#mat=(mat-Mmin)/(Mmax-Mmin)
mat=mat*2-1
game="perSoccer"
#print(mat)
thr=np.linalg.matrix_rank(mat)
print(thr)
alpha=0.001
alpha_rank_partial=partial(alpha_rank,
                                 alpha=alpha,
                                 mutation=50,
                                 use_inf_alpha=False,
                                 inf_alpha_eps=0.0001,
                                 use_sparse=False,
                                 use_cache=True)

true_alpha_rank=alpha_rank_partial(np.array([mat]))
'''
sortrank=-np.sort(-true_alpha_rank)
print(sortrank[:10])
print(sortrank[20:30])
print(sortrank[40:60])
print(sortrank[60:80])

print(sortrank[0]-sortrank[1])
print(sortrank[0]-sortrank[10])
if np.abs(sortrank[0]-sortrank[1])>1e-16:
    print(11111)
if np.abs(1-1)>1e-10:
    print(11111)
n=mat.shape[0]
print(np.min(sortrank))
'''
def MC(r,m):
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

                (X, S, Y) = optspace.optspace(smat, rank_n=r,
                                              num_iter=200000,
                                              tol=1e-4,
                                              verbosity=1,
                                              outfile=b""
                                              )

                [X, S, Y] = map(np.matrix, [X, S, Y])
                bM = X * S * Y.T

                mmse = np.sqrt(np.sum(np.power(bM - mat, 2)) / X.shape[0] / Y.shape[0])
                pi_r = alpha_rank_partial(np.array([bM]))
                pi_max = np.max(np.abs(pi_r - true_alpha_rank))
                rank_error = calrank(true_alpha_rank, pi_r,eps=1e-20)
                rank_error_10 =calrank(true_alpha_rank,pi_r,eps=1e-10)
                pbst = np.max(PBS(mat, true_alpha_rank))
                pbse = np.max(PBS(mat, pi_r))
                mse_pi_error = pbst - pbse
                end_time=time.time()
                print("MC algorithm time:",end_time-start_time)
                return [mmse,pi_max,rank_error,rank_error_10,mse_pi_error],end_time-start_time

def picture(m,X,label,linestyle):
    print(X)
    print(X.shape)
    rep=X.shape[0]
    Merror=X[:,0,:].reshape(rep,-1)
    pierror=X[:,1,:].reshape(rep,-1)
    rankerror=X[:,3,:].reshape(rep,-1)
    msepierror=X[:,4,:].reshape(rep,-1)
    X=np.array(m)
    se = scipy.stats.sem(Merror, axis=0)
    plt.subplot(221)

    plt.fill_between(X, Merror.mean(0) - se, Merror.mean(0) + se,
                     zorder=10, alpha=0.2)
    plt.plot(X, Merror.mean(0), label=label, zorder=10,linestyle=linestyle)
    # plt.plot(X[0], X[1],label=label)

    # ax1.set_title("M error")
    plt.subplot(222)

    se = scipy.stats.sem(pierror, axis=0)
    plt.fill_between(X, pierror.mean(0) - se, pierror.mean(0) + se,
                     alpha=0.2)
    plt.plot(X, pierror.mean(0), label=label,linestyle=linestyle)
    # ax2.set_title("C error")
    plt.subplot(223)

    se = scipy.stats.sem(rankerror, axis=0)
    plt.fill_between(X, rankerror.mean(0) - se, rankerror.mean(0) + se,
                     alpha=0.2)
    plt.plot(X, rankerror.mean(0), label=label,linestyle=linestyle)

    plt.subplot(224)
    msepierror=np.abs(msepierror)
    se = scipy.stats.sem(msepierror, axis=0)
    plt.fill_between(X, msepierror.mean(0) - se, msepierror.mean(0) + se,
                     alpha=0.2)
    plt.plot(X, msepierror.mean(0), label=label, linestyle=linestyle)


    #
def run_exp(d):
    i, exp = d
    exp_data={}
    exp_data["MCr"]=exp["MCr"]
    exp_data["MCm"]=exp["MCm"]
    exp_data["rep"]=exp["rep"]
    exp_data["data"],exp_data["time"]=MC(exp_data["MCr"],exp_data["MCm"])#[1,2,3]
    return exp_data

#if __name__=='__main__':
def run_game():
    base_params = {
        # Alpha rank hyper-parameters

        "MC": [True],
        "MCr": [1,2,4,8,10,16],#[10,12,15,20,25],
        #"MCm":[5000]+[ i for i in range(10000,100001,10000)],
        #"MCm": [i for i in range(10000, 610001, 100000)],
        "MCm":[i for i in range(5000,6000+1,1000)]+[i for i in range(8000,32000+1,4000)]#[2000,2500,3000,3500,4000,4500,5000,5500,6000,7000,8000,9000,9500]
    }
    exp_params = [{"rep":[1,2]}]
    exps=[]

    for exp_dict in exp_params:
        full_dict = {**base_params,**exp_dict}

        some_exps = list(map(dict, product(*[[(k, v) for v in vv] for k, vv in full_dict.items()])))
        exps = exps + some_exps
    print(len(exps))
    print(exps[0])
    num_in_parallel = 10#len(exps)
    start_time = time.time()
    print("--- Starting {} Experiments ---".format(len(exps)))
    r_line={}
    with Pool(num_in_parallel) as pool:
        ix = 0
        for exp_data in pool.map(run_exp, [(i, exp) for i, exp in enumerate(exps)]):
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
    with open("../picture_data/{}/{}/perSoccer200kfin10conv.pkl".format(alpha,game), "wb") as f:
        pickle.dump(r_line, f)
    for r in base_params["MCr"]:
        omega=[]
        for m in base_params["MCm"]:
            omega.append(r_line[r][m])
        omega=np.array(omega)
        omega=omega.transpose((1,2,0))
        if r==thr:
            picture(base_params["MCm"],omega,label="r={}".format(r),linestyle='--')
        else:
            picture(base_params["MCm"], omega, label="r={}".format(r), linestyle='-')

    #X=MC()
    #picture(X)
    plt.subplot(131)
    plt.grid()
    plt.subplot(132)
    plt.grid()
    plt.subplot(133)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig('../picture_data/{}/{}/perSoccer200kfin10conv.pdf'.format(alpha,game))
    plt.show()

if __name__=='__main__':
    print(mat)
    #run_game()


    with open("/data/yanxue/marleval/picture_data/{}/{}/perSoccer200kfin10conv.pkl".format(alpha,game), "rb") as f:
        r_line=pickle.load(f)
    print(r_line)
    rlist=[1,2,4,8,10,16]
    #[i for i in range(5000,6000+1,1000)]+\
    mlist =[4000,5000,6000]+[i for i in range(8000,32000+1,4000)]#[2000,2500,3000,3500,4000,4500,5000,5500,6000,7000,8000,9000,9500]#[2000,4000,6000]#[5000]+[i for i in range(10000,100001,10000)]
    print(mlist)
    for r in rlist:
        omega=[]
        for m in mlist:
            print(r,m,r_line[r][m])
            omega.append(r_line[r][m])
        omega=np.array(omega)
        omega=omega.transpose((1,2,0))
        if r == thr:
            picture(mlist, omega, label="r={}".format(r), linestyle='--')
        else:
            picture(mlist, omega, label="r={}".format(r), linestyle='-')

    #X=MC()
    #picture(X)
    plt.subplot(221)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.grid()
    plt.subplot(222)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.grid()
    plt.subplot(223)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.grid()
    plt.subplot(224)
    plt.xticks(fontsize=fz)
    plt.yticks(fontsize=fz)
    plt.grid()
    plt.tight_layout()
    plt.legend(fontsize=fz,loc='right')
    plt.savefig('/data/yanxue/marleval/picture_data/{}/final/perSoccer.pdf'.format(alpha))
    #plt.show()
    #plt.figure(figsize=(1, 6))

    #plt.figlegend(
    #    *axes[0].get_legend_handles_labels(),
    #    loc='center up')
    #
    #plt.show()
    #plt.savefig('/data/yanxue/marleval/picture_data/{}/{}/soccerlegend.pdf'.format(alpha,game))
    plt.show()

