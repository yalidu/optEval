import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = '/Users/lhchen/Downloads/'
file_lists = [['run-%i_.-tag-others_local_1_tb_rews.csv' % i for i in [199, ] + list(range(220, 229))],
         ['run-%i_.-tag-others_local_1_tb_rews.csv' % i for i in [214, ] + list(range(247, 253)) + list(range(254, 256))],
         ['run-%i_.-tag-others_local_1_tb_rews.csv' % i for i in [213, ] + list(range(238, 246))]]
name_lists = ['IND-RE vs IND-RE', 'SIC-RE vs IND-RE', 'SIC-RE vs SIC-RE']
color_lists = ['green', 'crimson', 'steelblue']

# n = 0
#
# for i in range(3):
#     file_list = file_lists[i]
#     value_list = []
#     for file in file_list:
#         with open(path+file, newline='') as csvfile:
#             reader = csv.reader(csvfile, delimiter=',', quotechar='|')
#             value_list.append([])
#             for line in reader:
#                 if line[1] == 'Step':
#                     continue
#                 step = int(line[1])
#                 value = float(line[2])
#                 value_list[-1].append(value)  # [n, dim]
#     # print(np.array(value_list).shape)
#     data = np.array(value_list)
#     # n = len(value_list)
#     # data = np.transpose(value_list, [1, 0])
#     # df = pd.DataFrame(data, index=['Number of episodes', 'Averaged reward']).melt()
#     df = pd.DataFrame(data).melt()
#     sns.lineplot(x='variable', y='value', data=df, label=name_lists[i], linewidth=2, err_style='band', color=color_lists[i])
# plt.legend(loc='lower right')
# plt.xlabel('x1k Iterations')
# plt.ylabel('Averaged rewards')
# n = len(value_list[-1])
# xs = np.arange(0, n+1, step=200)
# plt.ylim(-0.6, 0.6)
# plt.xticks(xs, map(str, xs//10))
# plt.savefig('/Users/lhchen/Desktop/matrix_game_curves.pdf')
# plt.show()

# data = [np.random.normal(loc=1, size=(10,)),
#         np.random.normal(loc=-1, size=(10,))]

data = [
    [3.22, 4, 2.88, 2.2338, 2.961, 1.8701, 3.0909, 3.5584, 3.7922, 3.0909],
    [3.480519, 4.779221, 3.402597, 2.831169, 3.194805, 2.675325, 4.077922, 3.74026, 3.402597, 2.779221],
    [3.376623, 3.74026, 3.142857, 2.935065, 2.987013, 2.805195, 3.298701, 3.818182, 4.207792, 3.220779],
    [4.103896, 4.051948, 3.012987, 2.311688, 3.61039, 3.324675, 3.532468, 3.298701, 3.376623, 3.402597333],
    [3.7403, 3.7922, 3.0649, 3.4805, 3.4805, 3.7922, 3.8442, 3.2468, 3.5065, 2.8571],
    [3.090909, 3.662338, 3.506494, 2.493506, 2.935065, 2.935065, 4.311688, 3.792208, 3.428571, 3.792208],
    [3.090909, 3.766234, 3.714286, 2.831169, 2.701299, 3.324675, 3.87013, 3.246753, 4.051948, 3.142857],
]

data = np.transpose(data, [1, 0])
df = pd.DataFrame(data).melt()
df.variable.replace({0: 0, 1: 5, 2: 10, 3: 15, 4: 20, 5: 25, 6:30}, inplace=True)
# sns.boxplot(x='variable', y='value', data=df, linewidth=2, color='orange')
sns.lineplot(x='variable', y='value', data=df, linewidth=2, color=[0.2, 0.6, 0.8], err_style='bars')
# plt.xticks(range(5), [0, 5, 10, 20, 30])
plt.ylabel('Averaged reward')
plt.xlabel('Signal dimension')
plt.tick_params()
# plt.grid()
plt.savefig('/Users/lhchen/Desktop/sensitivity.pdf')
plt.show()
