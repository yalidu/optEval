# alpha-IG

We compare our optEval with alpha-IG proposed in “Estimating alpha-Rank by Maximising Information Gain”. Paper is available [here](https://arxiv.org/abs/2101.09178). The code implementation is available in https://github.com/microsoft/InfoGainalpharank. 



## alpha-IG on Gaussian game 

To do experiments on Gaussian(15*15), please run the following instruction in file src/. 

```
python run_experiments.py 
```

We sightly change the source code of paper, where we load '/marl_eval/data/gaussian15.npy' in 'src/games/gaussian_one_pop' as game payoff, and we repeat this experiment for three times, and their mean value was taken as the final result.



