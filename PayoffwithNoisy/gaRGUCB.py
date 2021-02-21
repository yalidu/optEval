# coding=utf-8
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from concurrent.futures import ProcessPoolExecutor as Pool
from itertools import product
from alpha_rank import alpha_rank
from RGUCB import FreqBandit


import time
from functools import partial
import game
from sampling import run_sampling
from time import sleep
import pickle
import os
import random
import sys
import datetime
def get_mask(n,r,m):
    pairs = []

    for i in range(n):

        j = random.randint(0, n - 1)
        pairs.append((i, j))
        if i!=j:
            pairs.append((j,i))
    for j in range(n):

        i = random.randint(0, n - 1)
        if (i, j) not in pairs:
            pairs.append((i, j))
        if (j,i) not in pairs:
            pairs.append((j,i))
    rest = m - len(pairs)

    select = []
    for i in range(n):
        for j in range(n):
            if (i, j) not in pairs:
                select.append((i, j))
    if rest > 0:
        random.shuffle(select)
        rest = min(rest, len(select))
        index=0
        while(rest>0):
            while(select[index] in pairs):
                index+=1

            pairs.append(select[index])
            x=select[index][0]
            y=select[index][1]
            rest-=1
            if(y,x) not in pairs:
                pairs.append((y,x))
                rest-=1

    return pairs
# Base set of hyper-parameters are specified in this dictionary

#init_r=int(sys.argv[1])

base_params = {
    # Alpha rank hyper-parameters
    "alpha": [0.001],
    "alpha_mutation": [50],
    "inf_alpha": [False],
    "inf_alpha_eps": [0.00000001],
    "alpha_rank_cache": [True],
    "alpha_rank_sparse": [False],
    "MC":[False],
    "MCr":[1],
    "MCm":[1],#[45,50,75,90,100,120,150,175,200],#[i for i in range(2000,6000+1,1000)]+[i for i in range(8000,32000+1,4000)],
    #"MCm":[1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000],
    "payoff_distrib": ["indep_normal"],
    "noise_var": [1],
    "expected_hallucinate": [True],
    "expected_hallucinate_samples": [1],
    "hallucinate_samples": [1],
    "starting_mu": [0.5],
    "starting_var": [1],

    "sampler": ["freq2"],
     #"delta": [0.1],

    "mc_samples": [200],

    "env_seed": [10 + i for i in range(1)],

    #"env": ["{}_Agent_Ties_{}_hard".format(n, h) for n in [8] for h in [10]],
    #"min_payoff": [0],
    #"max_payoff": [1],

    #"repeats": [1 + i for i in range(3)],  # Number of repeats of the same experimental config
    "label": ["alphaRankExpsLabel"],  # Label to assign these experiments

    "t_max":[100*15*15],# [4000000],#[1000* 1000],  # Maximum timesteps to run for

    "graphing_samples": [2000],  # Number of alpha-ranks to save 100 times during training for graphing

}

# -- Setting the environment parameters -- #
# -- Comment/Uncomment appropriately -- #


env_params = {
    "env_seed": [10 + i for i in range(1)],
    "env":["bernoulli"],#: ["{}_Agent_Ties".format(8)],
    "min_payoff": [-1.2],
    "max_payoff": [1.2],
}

base_params = {**env_params, **base_params}
print(base_params)

# -- End of setting env parameters --

# Parameters for the algorithms that are going to be run
exp_params = [
    # ResponseGraphUCB with a range of \delta values
    {
        "sampler": ["freq2"],
        "delta": [0.01,0.1,0.2],#[0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001],
        "repeats": [1,2,3]#[1 + i for i in range(6)],
    }
]
print("--------------")
print(env_params)
env_dict = {}

env_dict["bernoulli"]=partial(game.GaussianOnePopGames,actions=15,rank=2)#partial(game.Bernoulli_game)
#env_dict['bad_good']=partial(game.BadAgentTies,actions=5,hardness=10)
num_in_parallel = 10  # Number of runs to launch in parallel
folder = "alphaRankRunsga15"  # Folder to save the experiments
SAVING = True  # If you are saving the results or not

exps = []
if exp_params == []:
    exp_params = [{}]
for exp_dict in exp_params:
    full_dict = {**base_params, **exp_dict}
    some_exps = list(map(dict, product(*[[(k, v) for v in vv] for k, vv in full_dict.items()])))
    exps = exps + some_exps


def run_exp(d):
    i, exp = d
    exp_start_time = time.time()
    print("Starting exp {}".format(i + 1))

    # Alpha rank function
    alpha_rank_partial = partial(alpha_rank,
                                 alpha=exp["alpha"],
                                 mutation=exp["alpha_mutation"],
                                 use_inf_alpha=exp["inf_alpha"],
                                 inf_alpha_eps=exp["inf_alpha_eps"],
                                 use_sparse=exp["alpha_rank_sparse"],
                                 use_cache=exp["alpha_rank_cache"])

    # Env
    payoffs = env_dict[exp["env"]]()#(seed=exp["env_seed"])
    num_pops, num_players, num_strats = payoffs.get_env_info()
    true_alpha_rank=alpha_rank_partial(payoffs.true_payoffs())
    #1,2,10
    #if exp["payoff_distrib"] == "indep_normal":
    tmax=exp["t_max"]

    mask=None
    if exp["MC"]==True:
        mask=get_mask(num_strats,exp["MCr"],exp["MCm"])
        tmax=len(mask)*100
    if exp["sampler"] == "freq2":  # ResponseGraphUCB
        sampler = FreqBandit(num_pops, num_strats, num_players, max_payoff=exp["max_payoff"],
                             min_payoff=exp["min_payoff"], delta=exp["delta"], alpha_rank_func=alpha_rank_partial,mask=mask)

    logging_info = run_sampling(payoffs,
                                sampler=sampler,
                                max_iters=tmax,#exp["t_max"],
                                graph_samples=exp["graphing_samples"],
                                true_alpha_rank=true_alpha_rank,
                                true_payoff=payoffs.true_payoffs(),
                                mask=mask,
                                r=exp["MCr"],
                                alpha_rank_func=alpha_rank_partial)

    exp_end_time = time.time()
    print("Finished exp {} in {:,} seconds".format(i + 1, exp_end_time - exp_start_time))

    logging_info["exp_info"] = exp
    logging_info["env_info"] = {
        "true_payoffs": payoffs.true_payoffs(),
        "num_pops": num_pops,
        "num_players": num_players,
        "num_strats": num_strats,
        "true_alpha_rank":true_alpha_rank
    }

    return logging_info

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print("Hi, {0}".format(name))  # Press ⌘F8 to toggle the breakpoint.


direc = "/data/yanxue/{}".format(folder)
if SAVING and not os.path.exists(direc):
    os.makedirs(direc, exist_ok=True)
print(direc)
date_str = "RGUCBdmk"#"RGUCB"#"A"+str(datetime.date.today())

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start_time = time.time()
    print("--- Starting {} Experiments ---".format(len(exps)))
    print(exps)
    if not SAVING:
        print("!!!!! DATA WILL NOT BE SAVED !!!!!")
    num_in_parallel=20#len(exps)#//3
    #sleep(10)
    with Pool(num_in_parallel) as pool:
        ix = 0
        for exp_data in pool.map(run_exp, [(i, exp) for i, exp in enumerate(exps)]):
            if SAVING:
                print("---SAVING DATA---")
                # Save the data in a dictionary
                print("Saving data for exp {}".format(ix + 1))
                # Save to a new file
                name_num = ix
                name = "{}_{}".format(date_str, name_num)
                print(direc+name)

                #while os.path.isfile("{}/{}.pkl".format(direc, name)):
                #    name_num += 1
                #    name = "{}_{}".format(date_str, name_num)

                with open("{}/{}.pkl".format(direc, name), "wb") as f:
                    pickle.dump(exp_data, f)
                print("save {}/{}".format(direc, name))
                #sleep(1)
            else:
                print("++++ NOT SAVING DATA FOR EXP {}++++".format(ix + 1))
            ix += 1


    end_time = time.time()
    print("Finished all {} experiments in {:,} seconds.".format(len(exps), end_time - start_time))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
