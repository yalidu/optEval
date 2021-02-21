# Estimating alpha-Rank from A Few Entries with Low Rank Matrix Completion

We propose two algorithms optEval-1 and optEval-2 calculating estimated alpha-rank based on matrix completion alogrithm  OptSpace under payoff matrices without and with noise respectively. We compare our optEval-2 with RGUCB in "Multiagent Evaluation under Incomplete Information" and alpha-IG in "Estimating alpha-Rank by Maximising Information Gain". 

Please install the environment before run experiments by running

```
pip install -r requirements.txt
```

## Data

We conduct experiments on Bernoulli gamers(100,100), Soccer-meta game(200,200), Gaussian game (15, 15) and 12 real world games including 'tic_tac_toe', 'AlphaStar', 'Kuhn-poker', etc. Payoff matrices we used in experiments are save in data/, and payoffs of real world games can be download at https://papers.nips.cc/paper/2020/file/ca172e964907a97d5ebd876bfdd4adbd-Supplemental.zip.

Statistical histograms of 28 real world games and line graphs of 5 games in them can be made by running 

```
python statistics.py
```



## OptSpace  and optEval-1

OptSpace is a matrix completion algorithm proposed in paper “OptSpace : A Gradient Descent Algorithm on the Grassman Manifold for Matrix Completion”  and details of algorithm OptSpace installation and the implemention of OptEval-1 can be find in file pyOptspace/



## optEval-2

The details of implementations of optEval-2 and RGUCB are described in file PayoffwithNoisy/



## Alpha-IG

We compare our optEval-2 with alpha-IG proposed in "Estimating alpha-Rank by Maximising Information Gain" on Gaussian game. Experiment setting are described in detail in file InfoGainalpharank/. 