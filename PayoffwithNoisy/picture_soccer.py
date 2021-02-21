import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
from matplotlib.ticker import StrMethodFormatter
import sys
#sys.path.append("../")
from alpha_rank import alpha_rank
from functools import partial
import scipy.stats
from alpha_rank import alpha_rank
alpha=0.001
alpha_rank_partial = partial(alpha_rank,
                                 alpha=alpha,
                                 mutation=50,
                                 use_inf_alpha=False,
                                 inf_alpha_eps=0.0001,
                                 use_sparse=False,
                                 use_cache=True)
eps=1e-20
def calrank(pi,pi_hat):
    #print(pi,pi_hat)
    n=pi.shape[0]
    rerror=0
    for i in range(n):
        for j in range(n):
            if i !=j:
                if pi[i]-pi[j]>eps:
                    if pi_hat[i]-pi_hat[j]<-eps:
                        rerror+=1
                else:
                    if pi[i] -pi[j]< -eps:
                        if pi_hat[i]-pi_hat[j]>eps:
                            rerror+=1
    return 1.0*rerror/n/n


fz=14
_, axes=plt.subplots(14,figsize=(8,6))

plt.subplot(221)

plt.xticks(fontsize=fz)
plt.yticks(fontsize=fz)
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
def picture(X,label,m,line='-'):
    rep=X.shape[0]
    Merror=X[:,0,:].reshape(rep,-1)
    pierror=X[:,1,:].reshape(rep,-1)
    rankerror=X[:,3,:].reshape(rep,-1)
    msepierror=X[:,2,:].reshape(rep,-1)
    msepierror=np.abs(msepierror)
    X=np.array(m)
    se = scipy.stats.sem(Merror, axis=0)
    plt.subplot(221)

    plt.fill_between(X, Merror.mean(0) - se, Merror.mean(0) + se,
                     zorder=10, alpha=0.2)
    plt.plot(X, Merror.mean(0), label=label, zorder=10,linestyle=line)
    plt.xlim(np.min(X), np.max(X))

    # plt.plot(X[0], X[1],label=label)
    plt.subplot(222)

    # ax1.set_title("M error")
    se = scipy.stats.sem(pierror, axis=0)
    plt.fill_between(X, pierror.mean(0) - se, pierror.mean(0) + se,
                     alpha=0.2)
    plt.plot(X, pierror.mean(0), label=label,linestyle=line)
    plt.xlim(np.min(X), np.max(X))

    # ax2.set_title("C error")
    plt.subplot(223)

    se = scipy.stats.sem(rankerror, axis=0)
    plt.fill_between(X, rankerror.mean(0) - se, rankerror.mean(0) + se,
                     alpha=0.2)
    plt.plot(X, rankerror.mean(0), label=label,linestyle=line)
    plt.xlim(np.min(X), np.max(X))

    plt.subplot(224)

    se = scipy.stats.sem(msepierror, axis=0)
    plt.fill_between(X, msepierror.mean(0) - se, msepierror.mean(0) + se, alpha=0.2)
    plt.plot(X, msepierror.mean(0), label=label,linestyle=line)
    plt.xlim(np.min(X), np.max(X))

def readRGB(game):
    RGUCBdict={}
    for i in range(9):
        d="/data/yanxue/marleval/marleval/picture_data/0.001/noisysoccer/alphaRankRuns2P1/"+"RGUCB_{}.pkl".format(i)
        with open(d, "rb") as f:
            data_dict = pickle.load(f)
            exp_info = data_dict['exp_info']

            repeats = exp_info['repeats']
            delta = exp_info['delta']

            if delta not in RGUCBdict.keys():
                RGUCBdict[delta] = []
            true_payoff = data_dict["env_info"]["true_payoffs"]
            true_payoff=2*true_payoff-1
            last_payoff = data_dict["last_payoff"]
            last_payoff=2*last_payoff-1
            merror = np.mean(np.square(true_payoff - last_payoff))
            true_alpha = alpha_rank_partial(true_payoff)
            last_alpha = alpha_rank_partial(last_payoff)
            #print(true_alpha)
            print("true")
            for i in range(100):
                if true_alpha[i] > 0.01:
                    print(i, true_alpha[i])
            #print(last_alpha)
            print("last")
            for i in range(100):
                if last_alpha[i] > 0.01:
                    print(i, last_alpha[i])

            pi_error = np.max(np.abs(true_alpha - last_alpha))
            pbst = np.max(PBS(true_payoff, true_alpha))
            pbse = np.max(PBS(true_payoff, last_alpha))
            mse_pi_error = pbst - pbse
            #mse_pi_error = np.mean(np.square(true_alpha - last_alpha))
            rank_error = calrank(true_alpha, last_alpha)
            RGUCBdict[delta].append([merror, pi_error, mse_pi_error, rank_error])

    print(RGUCBdict)
    return RGUCBdict
def readopt(r):
    optdict={}
    for i in range(60):
        d="/data/yanxue/marleval/marleval/picture_data/0.001/noisysoccer/alphaRankRuns2P1/"+"optevaltr{}_{}.pkl".format(r,i)
        with open(d, "rb") as f:
            data_dict = pickle.load(f)
            exp_info = data_dict['exp_info']

            repeats = exp_info['repeats']
            delta = exp_info['delta']

            r=exp_info['MCr']
            m=exp_info['MCm']

            if delta not in optdict.keys():
                optdict[delta] = {}
            if m not in optdict[delta].keys():
                optdict[delta][m]=[]

            true_payoff = data_dict["env_info"]["true_payoffs"]
            last_payoff = data_dict["last_payoff"]
            merror = np.mean(np.square(true_payoff - last_payoff))
            true_alpha = alpha_rank_partial(true_payoff)
            last_alpha = alpha_rank_partial(last_payoff)
            #print(true_alpha)
            '''
            print("true")
            for i in range(100):
                if true_alpha[i] > 0.01:
                    print(i, true_alpha[i])
            #print(last_alpha)
            print("last")
            for i in range(100):
                if last_alpha[i] > 0.01:
                    print(i, last_alpha[i])
            '''
            pi_error = np.max(np.abs(true_alpha - last_alpha))
            pbst = np.max(PBS(true_payoff, true_alpha))
            pbse = np.max(PBS(true_payoff, last_alpha))
            mse_pi_error = pbst - pbse
            #mse_pi_error = np.mean(np.square(true_alpha - last_alpha))
            rank_error = calrank(true_alpha, last_alpha)
            optdict[delta][m].append([merror, pi_error, mse_pi_error, rank_error])
    print(np.argsort(-true_alpha))
    print(np.argsort(-last_alpha))

    print("truep4,18,last 18,4",true_payoff[0][4][18],true_payoff[0][18][4])
    print("lastp4,18,last 18,4",last_payoff[0][4][18],last_payoff[0][18][4])
    print(optdict)
    return optdict
#get_data(d,"RGUCB",5)
game="noisysoccer200"

RGB_data=readRGB(game)
with open("./picture_data/{}/{}RGUCB.pkl".format(alpha,game), "wb") as f:
    pickle.dump(RGB_data, f)

opteval_data_11=readopt(r=1)
with open("./picture_data/{}/{}optevalr1.pkl".format(alpha,game), "wb") as f:
    pickle.dump(opteval_data_11, f)
opteval_data_15=readopt(r=2)
with open("./picture_data/{}/{}optevalr2.pkl".format(alpha,game), "wb") as f:
    pickle.dump(opteval_data_15, f)

opteval_data_20=readopt(r=4)
with open("./picture_data/{}/{}optevalr4.pkl".format(alpha,game), "wb") as f:
    pickle.dump(opteval_data_20, f)

opteval_data_25=readopt(r=8)
with open("./picture_data/{}/{}optevalr8.pkl".format(alpha,game), "wb") as f:
    pickle.dump(opteval_data_25, f)


opteval_data_20=readopt(r=10)
with open("./picture_data/{}/{}optevalr10.pkl".format(alpha,game), "wb") as f:
    pickle.dump(opteval_data_20, f)

opteval_data_25=readopt(r=16)
with open("./picture_data/{}/{}optevalr16.pkl".format(alpha,game), "wb") as f:
    pickle.dump(opteval_data_25, f)

'''

d="alphaRankRunssoccer200/"+"optevalr{}_{}.pkl".format(10,0)
with open(d, "rb") as f:
            data_dict = pickle.load(f)
            exp_info = data_dict['exp_info']

            repeats = exp_info['repeats']
            delta = exp_info['delta']

            r=exp_info['MCr']
            m=exp_info['MCm']
            print(r,m)
            true_payoff = data_dict["env_info"]["true_payoffs"]
            print("true_payoff:")
            print(true_payoff)
            last_payoff = data_dict["last_payoff"]
            print("last_payoff:")
            print(last_payoff)
            print(np.mean(np.square(true_payoff-last_payoff)))


d="alphaRankRunssoccer200/"+"optevalr{}_{}.pkl".format(10,0)
with open(d, "rb") as f:
            data_dict = pickle.load(f)
            exp_info = data_dict['exp_info']

            repeats = exp_info['repeats']
            delta = exp_info['delta']

            r=exp_info['MCr']
            m=exp_info['MCm']
            print(r,m)
            true_payoff = data_dict["env_info"]["true_payoffs"]
            print("true_payoff:")
            print(true_payoff)
            last_payoff = data_dict["last_payoff"]
            print("last_payoff:")
            print(last_payoff)
            print(np.mean(np.square(true_payoff-last_payoff)))

'''
delta=0.2
m=[i for i in range(4000,6000+1,1000)]+[i for i in range(8000,24000+1,4000)]#[1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500]
d="./picture_data/{}/{}RGUCB.pkl".format(alpha,game)

#d="./picture_data/{}RGUCB.pkl".format(game)

with open(d, "rb") as f:
    RGB_dict = pickle.load(f)
    print(RGB_dict[delta])
    RGB=np.array(RGB_dict[delta])
    RGB=RGB.reshape((3,4,1))
    RGB=np.repeat(RGB,len(m),axis=2)
    print(RGB)
    picture(RGB,label="RGUCB $\delta$={}".format(delta),m=m)

d="./picture_data/{}/{}optevalr1.pkl".format(alpha,game)
#d="./picture_data/{}optevalr10.pkl".format(game)

with open(d, "rb") as f:
    opt_dict = pickle.load(f)
    print(opt_dict[delta])
    opt=opt_dict[delta]
    result=[]#np.reshape(np.array(opt[m[0]]),(3,4,1))
    for i in range(len(m)):
        result.append(opt[m[i]])
    result=np.array(result)
    result=result.transpose((1,2,0))
    print(result[:,:,0])

    picture(result, label="opteval r=1", m=m)
d="./picture_data/{}/{}optevalr2.pkl".format(alpha,game)
#d="./picture_data/{}optevalr15.pkl".format(game)

with open(d, "rb") as f:
    opt_dict = pickle.load(f)
    print(opt_dict[delta])
    opt=opt_dict[delta]
    result=[]#np.reshape(np.array(opt[m[0]]),(3,4,1))
    for i in range(len(m)):
        result.append(opt[m[i]])
    result=np.array(result)
    result=result.transpose((1,2,0))
    print(result[:,:,0])

    picture(result, label="opteval r=2", m=m)

d="./picture_data/{}/{}optevalr4.pkl".format(alpha,game)
#d="./picture_data/{}optevalr20.pkl".format(alpha,game)

with open(d, "rb") as f:
    opt_dict = pickle.load(f)
    print(opt_dict[delta])
    opt=opt_dict[delta]
    result=[]#np.reshape(np.array(opt[m[0]]),(3,4,1))
    for i in range(len(m)):
        result.append(opt[m[i]])
    result=np.array(result)
    result=result.transpose((1,2,0))
    print(result[:,:,0])

    picture(result, label="opteval r=4", m=m)
d="./picture_data/{}/{}optevalr8.pkl".format(alpha,game)
#d="./picture_data/{}optevalr25.pkl".format(alpha,game)

with open(d, "rb") as f:
    opt_dict = pickle.load(f)
    print(opt_dict[delta])
    opt=opt_dict[delta]
    result=[]#np.reshape(np.array(opt[m[0]]),(3,4,1))
    for i in range(len(m)):
        result.append(opt[m[i]])
    result=np.array(result)
    result=result.transpose((1,2,0))
    print(result[:,:,0])

    picture(result, label="opteval r=8", m=m)

d="./picture_data/{}/{}optevalr10.pkl".format(alpha,game)
#d="./picture_data/{}optevalr25.pkl".format(alpha,game)

with open(d, "rb") as f:
    opt_dict = pickle.load(f)
    print(opt_dict[delta])
    opt=opt_dict[delta]
    result=[]#np.reshape(np.array(opt[m[0]]),(3,4,1))
    for i in range(len(m)):
        result.append(opt[m[i]])
    result=np.array(result)
    result=result.transpose((1,2,0))
    print(result[:,:,0])

    picture(result, label="opteval r=10", m=m,line='--')

d = "./picture_data/{}/{}optevalr16.pkl".format(alpha, game)
# d="./picture_data/{}optevalr25.pkl".format(alpha,game)

with open(d, "rb") as f:
    opt_dict = pickle.load(f)
    print(opt_dict[delta])
    opt = opt_dict[delta]
    result = []  # np.reshape(np.array(opt[m[0]]),(3,4,1))
    for i in range(len(m)):
        result.append(opt[m[i]])
    result = np.array(result)
    result = result.transpose((1, 2, 0))
    print(result[:, :, 0])

    picture(result, label="opteval r=16", m=m)

#d="alphaRankRuns/"+"opteval14"
#get_data_o(d,"opteval rank=10 m=4000",6)
'''
d="alphaRankRuns/"+"opteval15"
get_data_o(d,"opteval rank=10 m=5000",6)
d="alphaRankRuns/"+"opteval13"
get_data_o(d,"opteval rank=10 m=3000",6)
'''
plt.subplot(2,2,1)
plt.grid()
plt.legend(loc='right up')

plt.tight_layout()

plt.subplot(2,2,2)
plt.grid()
plt.tight_layout()
plt.subplot(2,2,3)
plt.grid()

plt.tight_layout()
plt.subplot(2,2,4)
plt.grid()

plt.tight_layout()
#fig.title("delta=0.1")
plt.savefig('/data/yanxue/marleval/noisysoccer/{}/soccerdelta{}'.format(alpha,delta).replace('.','')+'.pdf')
plt.show()
