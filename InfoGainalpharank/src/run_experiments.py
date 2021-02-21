from itertools import product
from functools import partial
from games.gaussian_one_pop import GaussianOnePopGames
from games.bad_agent_ties import BadAgentTies
from sampling_schemes.distributions.indep_normal import IndependentNormal
from sampling_schemes.distributions.indep_normal_kernel import NormalKernel
from sampling_schemes.bayesian_bandit import BayesianBandit
from sampling_schemes.freq_bandit import FreqBandit
from sampling_schemes.random import RandomSampler
from sampling_schemes.payoff_bayesian_bandit import PayoffBayesianBandit
from sampling import run_sampling
from alpha_rank import alpha_rank
import os
import time
from time import sleep
import datetime
import pickle
from concurrent.futures import ProcessPoolExecutor as Pool
import json

# Script to setup, run and save experimental results

# Base set of hyper-parameters are specified in this dictionary
base_params = {
    # Alpha rank hyper-parameters
    "alpha": [0.001],
    "alpha_mutation": [50],
    "inf_alpha": [False],
    "inf_alpha_eps": [0.000001],
    "alpha_rank_cache": [True],
    "alpha_rank_sparse": [False],

    "payoff_distrib": ["indep_normal"],
    "noise_var": [1],

    "expected_hallucinate": [True],
    "expected_hallucinate_samples": [1],
    "hallucinate_samples": [1],
    "starting_mu": [0.5],
    "starting_var": [1],

    "sampler": ["bayesian"],
    "delta": [0.1],

    "mc_samples": [200],

    "env_seed": [10 + i for i in range(1)],
    "env": ["Gaussian_4_1pop"],  # .format(n, h) for n in [8] for h in [10]],
    "min_payoff": [-1.2],
    "max_payoff": [1.2],

    "repeats": [1,2,3],  # [1+i for i in range(3)], # Number of repeats of the same experimental config
    "label": ["alphaRankExpsLabel"],  # Label to assign these experiments

    "t_max": [100*15*15],  # Maximum timesteps to run for

    "graphing_samples": [2000],  # Number of alpha-ranks to save 100 times during training for graphing

}

# -- Setting the environment parameters -- #
# -- Comment/Uncomment appropriately -- #

# # Running on 4x4 Gaussian Games
env_params = {
    "env_seed": [10 + i for i in range(1)],
    "env": ["Gaussian_4_1pop"],
    "min_payoff": [-1.2],
    "max_payoff": [1.2],
}

# # Running on 2 Good, 2 Bad
# env_params = {
#     "env_seed": [10 + i for i in range(1)],
#     "env": ["{}_Agent_Ties".format(4)],
#     "min_payoff": [0],
#     "max_payoff": [1],
# }


# Running on 3 Good, 5 Bad

base_params = {**env_params, **base_params}
# -- End of setting env parameters --

# Parameters for the algorithms that are going to be run
exp_params = [

    # \alphaIG
    {
        "payoff_distrib": ["indep_normal"],
        "expected_hallucinate": [True],
        "expected_hallucinate_samples": [10],
        "hallucinate_samples": [100],
        "starting_var": [1],
        "noise_var": [0.5],
        "mc_samples": [500],
        "acquisition": ["entropy_support"],
        "repeat_sampling": [500],
    }

    # \alphaIG and \alphaWass with prior info
    # {
    #     "payoff_distrib": ["normal_kernel"],
    #     "expected_hallucinate": [True],
    #     "expected_hallucinate_samples": [10],
    #     "hallucinate_samples": [100],
    #     "starting_var": [1],
    #     "noise_var": [0.5],
    #     "mc_samples": [500],
    #     "acquisition": ["l1_relative", "entropy_support"],
    #     "repeat_sampling": [100],
    # },

]

num_in_parallel = 1  # Number of runs to launch in parallel
folder = "alphaRankRuns"  # Folder to save the experiments
SAVING = True  # If you are saving the results or not

exps = []
if exp_params == []:
    exp_params = [{}]
for exp_dict in exp_params:
    full_dict = {**base_params, **exp_dict}
    some_exps = list(map(dict, product(*[[(k, v) for v in vv] for k, vv in full_dict.items()])))
    exps = exps + some_exps

direc = "{}/{}".format(os.getcwd(), folder)
if SAVING and not os.path.exists(direc):
    os.makedirs(direc, exist_ok=True)



date_str = "alphaIGga15"#str(datetime.date.today())

start_time = time.time()

env_dict = {}
for i in range(5, 6, 1):
    env_dict["Gaussian_4_1pop"] = partial(GaussianOnePopGames, actions=15)
    # env_dict["{}_Agent_Ties".format(i)] = partial(BadAgentTies, actions=i, hardness=10)


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
    payoffs = env_dict[exp["env"]](seed=exp["env_seed"])
    num_pops, num_players, num_strats = payoffs.get_env_info()

    # Distribution over payoffs
    if exp["payoff_distrib"] == "indep_normal":
        payoff_distrib = IndependentNormal(num_pops, num_strats, num_players,
                                           starting_mu=exp["starting_mu"],
                                           starting_var=exp["starting_var"],
                                           noise_var=exp["noise_var"],
                                           hallucination_samples=exp["hallucinate_samples"])
    elif exp["payoff_distrib"] == "normal_kernel":
        payoff_distrib = NormalKernel(num_pops, num_strats, num_players,
                                      starting_mu=exp["starting_mu"],
                                      starting_var=exp["starting_var"],
                                      noise_var=exp["noise_var"],
                                      hallucination_samples=exp["hallucinate_samples"])

    if exp["sampler"] == "bayesian":  # \alphaIG and \alphaWass
        sampler = BayesianBandit(num_pops, num_strats, num_players,
                                 payoff_distrib=payoff_distrib,
                                 alpha_rank_func=alpha_rank_partial,
                                 mc_samples=exp["mc_samples"],
                                 acquisition=exp["acquisition"],
                                 expected_hallucinate=exp["expected_hallucinate"],
                                 expected_samples=exp["expected_hallucinate_samples"],
                                 use_parallel=True,
                                 repeat_sampling=exp["repeat_sampling"])
    elif exp["sampler"] == "payoff_bandit":  # Payoff
        sampler = PayoffBayesianBandit(num_pops, num_strats, num_players, payoff_distrib=payoff_distrib,
                                       alpha_rank_func=alpha_rank_partial)
    elif exp["sampler"] == "freq2":  # ResponseGraphUCB
        sampler = FreqBandit(num_pops, num_strats, num_players, max_payoff=exp["max_payoff"],
                             min_payoff=exp["min_payoff"], delta=exp["delta"], alpha_rank_func=alpha_rank_partial)
    elif exp["sampler"] == "random":  # Random
        sampler = RandomSampler(num_pops, num_strats, num_players, alpha_rank_func=alpha_rank_partial)

    logging_info = run_sampling(payoffs, sampler, max_iters=exp["t_max"], graph_samples=exp["graphing_samples"])

    exp_end_time = time.time()
    print("Finished exp {} in {:,} seconds".format(i + 1, exp_end_time - exp_start_time))

    logging_info["exp_info"] = exp
    logging_info["env_info"] = {
        "true_payoffs": payoffs.true_payoffs(),
        "num_pops": num_pops,
        "num_players": num_players,
        "num_strats": num_strats
    }

    return logging_info


print("--- Starting {} Experiments ---".format(len(exps)))
if not SAVING:
    print("!!!!! DATA WILL NOT BE SAVED !!!!!")

sleep(10)

with Pool(num_in_parallel) as pool:
    ix = 0
    print(exps)
    for exp_data in pool.map(run_exp, [(i, exp) for i, exp in enumerate(exps)]):
        if SAVING:
            print("---SAVING DATA---")
            # Save the data in a dictionary
            print("Saving data for exp {}".format(ix + 1))
            # Save to a new file
            name_num = ix
            name = "{}_{}".format(date_str, name_num)
            while os.path.isfile("{}/{}.pkl".format(direc, name)):
                name_num += 1
                name = "{}_{}".format(date_str, name_num)
            with open("{}/{}.pkl".format(direc, name), "wb") as f:
                pickle.dump(exp_data, f)
            sleep(1)
        else:
            print("++++ NOT SAVING DATA FOR EXP {}++++".format(ix + 1))
        ix += 1

end_time = time.time()
print("Finished all {} experiments in {:,} seconds.".format(len(exps), end_time - start_time))




