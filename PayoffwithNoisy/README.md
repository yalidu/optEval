# OptEval2-payoff matrices with noisy 

In game.py we set up several environment classes to load payoff matrices of  games and simulate independent repeat experiment.RGUCB.py implements the RGUCB algorithm proposed in ‘Multiagent Evaluation under Incomplete Information’. Sampling.py is used to obtain the estimated payoff matrix and calculate the matrix after matrix completion by running optSpace. The parameters of OptEval-2 algorithm are set in mainxx.py, and the RGUCB‘s parameters are set in xxRGUCB.py. Finally, run picturexx.py to get a visualization of the results.

## Bernoulli noisy game

Please run the following instructions to get the visualization of Bernoulli game with noisy payoff.

```
sh nb.sh 
python mainnb.py 1
python mainnb.py 2
python mainnb.py 4
python mainnb.py 8
python mainnb.py 10
python mainnb.py 16
python nbRGUCB.py
python picture.py
```

## Soccer noisy game

Please run the following instructions to get the visualization of soccer-meta game with noisy payoff.

```
sh ns.sh
python mainns.py 1
python mainns.py 2
python mainns.py 4
python mainns.py 8
python mainns.py 10
python mainns.py 16
python nsRGUCB.py
python picture_soccer.py
```

## Gaussian(15*15 rank=2) noisy game

Please run the following instructions to get the visualization of gaussian game with noisy payoff.

```
sh nga.sh
python mainga.py 1
python mainga.py 2
python mainga.py 4
python gaRGUCB.py
python picturega.py
```

## Comparisons of empirical sampling complexity and theoretical results

We random twenty games with n = 10, 20, ..., 200, r = 5 to compare the sampling complexity and theoretical results. 

We generate 20 games by

```
python random_game.py
```

### Noisy-free payoff

Compare samples m of optEval-1 with C*nrlogn, where C=0.4, 0.5, 0.6. 

```
python 20conv.py
```

### Noisy payoff  

Compare samples m of optEval-2 under noisy setting with C*nrlogn, where C=0.6, 0.8, 1.0. 

mainconv.py get the results of RGUCB with n=10, 20, ... , 200 as parameter. We repeat RGUCB three times and get their mean by 20noisyRGUCB_process.py. optconv.py get the results of RGUCB with n=10, 20, ... , 200 as parameter. Finally we get the comparison with convergence of the m and O(nrlogn) by running picture_20conv_noisy.py. 

Please run 'sh optconv.sh' to get the final visualization result.

```
python mainconv.py 10
python mainconv.py 20
python mainconv.py 30
python mainconv.py 40
python mainconv.py 50
python mainconv.py 60
python mainconv.py 70
python mainconv.py 80
python mainconv.py 90
python mainconv.py 100
python mainconv.py 110
python mainconv.py 120
python mainconv.py 130
python mainconv.py 140
python mainconv.py 150
python mainconv.py 160
python mainconv.py 170
python mainconv.py 180
python mainconv.py 190
python mainconv.py 200
python 20noisyRGUCB_process.py
python optconv.py 10
python optconv.py 20
python optconv.py 30
python optconv.py 40
python optconv.py 50
python optconv.py 60
python optconv.py 70
python optconv.py 80
python optconv.py 90
python optconv.py 100
python optconv.py 110
python optconv.py 120
python optconv.py 130
python optconv.py 140
python optconv.py 150
python optconv.py 160
python optconv.py 170
python optconv.py 180
python optconv.py 190
python optconv.py 200

python picture_20conv_noisy.py
```