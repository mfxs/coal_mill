# 导入库
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 忽略警告并设置画风
warnings.filterwarnings('ignore')
plt.style.use('seaborn-dark')


# Adjacency matrix
def plot_adj(data):
    sns.heatmap(pd.DataFrame(data, index=predict_variable, columns=predict_variable), 0, 1, 'YlGnBu', annot=True,
                fmt='.2f', annot_kws={'size': 15, 'weight': 'bold'})
    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')
    plt.xticks(fontsize=15, fontweight='black')
    plt.yticks(fontsize=15, fontweight='black', rotation='horizontal')


# R2&RMSE
def plot_r2_rmse(data, label, legend):
    pd.DataFrame(data.T, index=predict_variable).plot.bar()
    plt.xlabel('Predicted Variable', fontsize=25, fontweight='black')
    plt.ylabel(label, fontsize=25, fontweight='black')
    plt.legend(legend, fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20, fontweight='black', rotation='horizontal')
    plt.yticks(fontsize=20, fontweight='black')


# Loss
def plot_loss(data, legend):
    plt.figure()
    data = pd.DataFrame(np.stack(data.values()))
    for i in range(data.shape[0]):
        plt.plot(range(1, data.shape[1] + 1), data.iloc[i, :], lw=3)
    plt.xlabel('Epoch', fontsize=25, fontweight='black')
    plt.ylabel('Loss', fontsize=25, fontweight='black')
    plt.legend(legend, fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20, fontweight='black')
    plt.yticks(fontsize=20, fontweight='black')
    plt.xlim(1, data.shape[1])


# Predict&Test
def predict_test(exp, model):
    plt.figure()
    plt.plot(y_test, lw=3)
    plt.plot(y_pred[exp - 1][model], lw=3)
    plt.xlabel('Sample', fontsize=10, fontweight='black')
    plt.ylabel('Value', fontsize=10, fontweight='black')
    plt.grid()
    plt.xticks(fontsize=10, fontweight='black')
    plt.yticks(fontsize=10, fontweight='black')
    plt.xlim(0, y_test.shape[0])


# 导入数据
path = 'Results/'
results = np.load(path + 'results.npy', allow_pickle=True).item()
adj = results['adj']
r2 = results['r2']
rmse = results['rmse']
loss_hist = results['loss_hist']
y_pred = results['prediction']
predict_variable = ['quality_variable_1', 'quality_variable_2', 'quality_variable_3', 'quality_variable_4',
                    'quality_variable_5']

# 实验结果平均
n_exp = len(r2)
r2_avg = np.stack(r2[0].values())
rmse_avg = np.stack(rmse[0].values())
for i in range(1, n_exp):
    r2_avg += np.stack(r2[i].values())
    rmse_avg += np.stack(rmse[i].values())
r2_avg /= n_exp * 0.01
rmse_avg /= n_exp

# Plot
# plot_adj(adj[0]['gclstm'].cpu().detach().numpy())
# plot_loss(loss[0], list(map(lambda x: x.upper(), list(loss[0].keys()))))
# plot_r2_rmse(r2_avg, 'R2', list(map(lambda x: x.upper(), list(r2[0].keys()))))
# plot_r2_rmse(rmse_avg, 'RMSE', list(map(lambda x: x.upper(), list(rmse[0].keys()))))
# predict_test(5, 'mcgcn')
# plt.show()

# 存储数据
np.savetxt(path + 'r2.csv', r2_avg, delimiter=',')
np.savetxt(path + 'rmse.csv', rmse_avg, delimiter=',')
