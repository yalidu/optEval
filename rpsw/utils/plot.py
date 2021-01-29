import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd


# curve_color_list = ['cyan', 'red', 'green', 'orange', ]
#
# idx = np.arange(0, 5)
# mean_curve = np.array([12.28, 13.72, 16.56, 10.82, 8.86])
# b_curve = np.array([2.08, 1.63, 1.50, 1.35, 0.89])
#
# plt.fill_between(idx, mean_curve+b_curve, mean_curve-b_curve,
#                  facecolor='orange', alpha=0.3)
# plt.plot(idx, mean_curve, color='orange', label='SIC-MA',
#          linewidth=3)
# # plt.fill_between(idx, [12.11+1.39,]*5, [12.11-1.39,]*5, facecolor=curve_color_list[2], alpha=0.3)
# plt.plot(idx, [12.11,]*5, color='steelblue',
#          linestyle='dashed', label='MADDPG', linewidth=3)
#
# plt.xticks(idx, ['0', '1e-5', '1e-4', '1e-3', '1e-2'], fontsize=15)
# plt.yticks(fontsize=15)
# plt.ylabel('Mean Rewards', fontsize=15)
# plt.xlabel('Alpha', fontsize=15)
# plt.legend(fontsize=25)
# plt.savefig('/Users/lhchen/Desktop/alpha.pdf', dpi=1)
# plt.show()

# idx = np.arange(0, 6)
# mean_curve = np.array([12.11, 13.34, 13.65, 14.16, 16.56, 15.38])
# b_curve = np.array([1.39, 1.39, 1.74, 2.14, 1.50, 1.32])
#
# plt.fill_between(idx, mean_curve+b_curve, mean_curve-b_curve, facecolor='orange', alpha=0.3)
# plt.plot(idx, mean_curve, color='orange', label='SIC-MA vs MADDPG', linewidth=3)
#
# plt.xticks(idx, ['0', '5', '10', '15', '20', '25'], fontsize=15)
# plt.yticks(fontsize=15)
# plt.ylabel('Mean Rewards', fontsize=15)
# plt.xlabel('Dimension of Signal', fontsize=15)
# plt.savefig('/Users/lhchen/Desktop/sig_size.pdf')
# plt.show()

data=[6.7, 1.5, 12.11, 11.48, 2.05, 16.56, 17.31]
df = pd.DataFrame(data)
print(df)
sns.barplot(x='variable', y='value', data=df)
plt.show()
