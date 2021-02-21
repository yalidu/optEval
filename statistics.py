import matplotlib.pyplot as plt
import pickle
import numpy as np
import math
with open("/data/spinning_top_payoffs.pkl", "rb") as fh:
  payoffs = pickle.load(fh)
games=[]
h=[]




'''
for game_name in payoffs:


  print(f"Game name: {game_name}")
  n=payoffs[game_name].shape[0]
  #if n==3:
  #    continue
  print(f"Number of strategies: {n}")
  rank=np.linalg.matrix_rank(payoffs[game_name])
  print("rank:",rank)
  print()
  number=math.ceil(n*0.15)
  U, s, V = np.linalg.svd(payoffs[game_name])

  games.append((game_name,n,rank,s[:number].sum()/s.sum()))
  h.append(games[-1][3])
pickle.dump(h,open("hist.pkl","wb"))

'''

plt.figure(figsize=(4,3))
game_list=['AlphaStar', 'Kuhn-poker', 'tic_tac_toe', 'Blotto', 'Disc game']
X=np.arange(0,101,2)
X=X/100.0
print(X[-1])
for game_name in game_list:
    U, s, V = np.linalg.svd(payoffs[game_name])
    mat=payoffs[game_name]
    n=mat.shape[0]
    Y=[0]
    sum=0
    total=s.sum()
    for i in range(1,X.shape[0]):
        index=math.ceil(X[i]*n)
        print(n,X[i],index)

        Y.append(s[:index].sum()/total)

    
    #for i in range(mat.shape[0]):
    #    sum+=s[i]
    #    Y.append(sum/total)
 

    if game_name=='AlphaStar' or game_name=='Kuhn-poker' or game_name=='tic_tac_toe':
        plt.plot(X,Y,label=game_name,linestyle='--',linewidth = '2')
    else:
        plt.plot(X, Y, label=game_name,linewidth = '2')
plt.legend(fontsize=10)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig('/data/yanxue/alphaRankRuns/5games4310.pdf')
plt.show()
plt.close()




#games.sort(key=lambda x: x[3])
h=pickle.load(open("hist.pkl","rb"))

print(games)
part=[0,0,0,0,0,0,0,0,0]
for i in range(len(games)):
    part[math.floor(games[i][3]*8)]+=1

name_list = [0,1/8,2/8,3/8,4/8,5/8,6/8,7/8,1]
num_list = part
fig=plt.figure(figsize=(4,3))
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.hist(h,bins=name_list,edgecolor='silver',facecolor='dodgerblue',linewidth=1.5)#,facecolor='blue')
#plt.legend()
plt.grid(axis='y',color= 'gray',alpha = 0.4)



plt.ylim(0,8)
plt.tight_layout()
plt.savefig('/data/yanxue/alphaRankRuns/28gamehist4310.pdf')
#plt.bar(range(len(num_list)), num_list,tick_label = name_list)
plt.show()
plt.close()
