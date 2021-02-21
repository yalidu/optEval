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
plt.plot(X, X * 5 * np.log(X), label='nrlogn')
plt.plot(X, 0.8 * X * 5 * np.log(X), label='0.8nrlogn')

plt.plot(X,0.6*X*5*np.log(X),label='0.6nrlogn')




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
plt.savefig("nrlogn3.pdf")

plt.show()


