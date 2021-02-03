# 导入库
import os
import time
import joblib
import argparse
import warnings
import pandas as pd

from package import *
from sklearn.metrics import r2_score
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process.kernels import DotProduct
from sklearn.gaussian_process import GaussianProcessRegressor

# 忽略警告
warnings.filterwarnings('ignore')

# 参数解析
parser = argparse.ArgumentParser(description='Coal Mill Experiment')

# 实验设置参数
parser.add_argument('--length', type=int, default=5000)
parser.add_argument('--train_size', type=float, default=0.8)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_exp', type=int, default=1)

# PLSR参数
parser.add_argument('--n_components', type=int, default=3)

# LS-SVR参数
parser.add_argument('-c', type=float, default=100)
parser.add_argument('--sigma', type=float, default=300)

# GPR参数
parser.add_argument('--length_scale', type=float, default=400)
parser.add_argument('--alpha', type=float, default=1)

# 网络模型参数
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=2.5)
parser.add_argument('--step_size', type=int, default=50)
parser.add_argument('--gamma', type=float, default=0.5)

# 序列模型参数
parser.add_argument('--seq_len', type=int, default=50)

# 图模型参数
parser.add_argument('--graph_reg', type=float, default=0.05)
parser.add_argument('--self_con', type=float, default=0.2)


# 主函数
def main():
    args = parser.parse_args()

    # 模型选择及输入参数
    model_name = {1: 'PLSR', 2: 'LS-SVR', 3: 'GPR', 4: 'FCN', 5: 'LSTM', 6: 'GCN', 7: 'MC-GCN', 8: 'GC-LSTM'}
    print(model_name)
    model_select = list(input('Select models:'))

    # 初始化结果
    results = {'adj': [], 'r2': [], 'rmse': [], 'loss_hist': [], 'prediction': []}
    # os.mkdir('Results')
    f = open('Results/params.txt', 'w+')
    f.write('Parameters setting:\n{}\n\n'.format(args.__dict__))

    # 导入数据
    data = pd.read_excel('8号机磨煤机C_正常.xlsx', index_col=0, header=1, nrows=args.length + 5001)
    data = data.iloc[5001:, :]

    # 数据划分
    predict_variable = [3, 12, 15, 20, 23]
    y = data.iloc[:, predict_variable]
    X = data.drop(columns=y.columns)
    X_train, y_train = X.iloc[:int(args.length * args.train_size)], y.iloc[:int(args.length * args.train_size)]
    X_test, y_test = X.iloc[int(args.length * args.train_size):], y.iloc[int(args.length * args.train_size):]

    # 导出数据
    # X_train.to_csv('Results/X_train.csv', header=False, index=False)
    # X_test.to_csv('Results/X_test.csv', header=False, index=False)
    # y_train.to_csv('Results/y_train.csv', header=False, index=False)
    # y_test.to_csv('Results/y_test.csv', header=False, index=False)

    # 设定种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 多次实验
    for exp in range(args.n_exp):
        print('=====Experiment({}/{})====='.format(exp + 1, args.n_exp))
        f.write('=====Experiment({}/{})=====\n'.format(exp + 1, args.n_exp))
        results['adj'].append({})
        results['r2'].append({})
        results['rmse'].append({})
        results['loss_hist'].append({})
        results['prediction'].append({})

        # PLSR
        if '1' in model_select:
            flag = 1
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            reg = PLSRegression(args.n_components).fit(X_train, y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

        # LS-SVR
        if '2' in model_select:
            flag = 2
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            reg = LssvrModel(args.c, args.sigma).fit(X_train, y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

        # GPR
        if '3' in model_select:
            flag = 3
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            kernel = DotProduct() * RBF(args.length_scale, (args.length_scale, args.length_scale))
            reg = GaussianProcessRegressor(kernel=kernel, alpha=args.alpha).fit(X_train, y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

        # FCN
        if '4' in model_select:
            flag = 4
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            reg = FcnModel(X_train.shape[1], y_train.shape[1], (1024, 256, 256, 256), args.n_epoch, args.batch_size,
                           args.lr, args.weight_decay, args.step_size, args.gamma).fit(X_train, y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            results['loss_hist'][-1].update({model_name[flag]: reg.loss_hist})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

        # LSTM
        if '5' in model_select:
            flag = 5
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            reg = LstmModel(X_train.shape[1], y_train.shape[1], (1024,), (256, 256, 256), args.seq_len, args.n_epoch,
                            args.batch_size, args.lr, args.weight_decay, args.step_size, args.gamma).fit(X_train,
                                                                                                         y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            results['loss_hist'][-1].update({model_name[flag]: reg.loss_hist})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

        # GCN
        if '6' in model_select:
            flag = 6
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            reg = GcnModel(X_train.shape[1], y_train.shape[1], (1024,), (256, 256, 256), args.graph_reg, args.self_con,
                           args.n_epoch, args.batch_size, args.lr, args.weight_decay, args.step_size, args.gamma).fit(
                X_train, y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            results['loss_hist'][-1].update({model_name[flag]: reg.loss_hist})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

        # MC-GCN
        if '7' in model_select:
            flag = 7
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            reg = McgcnModel(X_train.shape[1], (1024,), (256,), (256, 256), y_train.shape[1], args.graph_reg,
                             args.self_con, args.n_epoch, args.batch_size, args.lr, args.weight_decay, args.step_size,
                             args.gamma).fit(X_train, y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['adj'][-1].update({model_name[flag]: reg.adj})
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            results['loss_hist'][-1].update({model_name[flag]: reg.loss_hist})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

        # GC-LSTM
        if '8' in model_select:
            flag = 8
            print('====={}====='.format(model_name[flag]))
            f.write('====={}=====\n'.format(model_name[flag]))

            # 训练&测试
            t1 = time.time()
            reg = GclstmModel(X_train.shape[1], (1024,), (256,), (256, 256), y_train.shape[1], args.seq_len,
                              args.graph_reg, args.self_con, args.n_epoch, args.batch_size, args.lr, args.weight_decay,
                              args.step_size, args.gamma).fit(X_train, y_train)
            t2 = time.time()
            y_pred = reg.predict(X_test)
            t3 = time.time()
            y_fit = reg.predict(X_train)
            print(reg.get_params())
            print('Time:\nFit: {:.3f}s Pred: {:.3f}s'.format(t2 - t1, t3 - t2))
            print('R2:\nFit: {} Pred: {}'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                 r2_score(y_test, y_pred, multioutput='raw_values')))

            # 写入文件
            f.write(str(reg.get_params()) + '\n')
            f.write('Time:\nFit: {:.3f}s Pred: {:.3f}s\n'.format(t2 - t1, t3 - t2))
            f.write('R2:\nFit: {} Pred: {}\n'.format(r2_score(y_train, y_fit, multioutput='raw_values'),
                                                     r2_score(y_test, y_pred, multioutput='raw_values')))

            # 存储结果和模型
            index = r2_rmse(y_test, y_pred, y.columns, f)
            results['adj'][-1].update({model_name[flag]: reg.adj})
            results['r2'][-1].update({model_name[flag]: index[0]})
            results['rmse'][-1].update({model_name[flag]: index[1]})
            results['prediction'][-1].update({model_name[flag]: y_pred})
            results['loss_hist'][-1].update({model_name[flag]: reg.loss_hist})
            joblib.dump(reg, 'Results/{}-{}.model'.format(model_name[flag], exp + 1))

    # 存储结果
    np.save('Results/results.npy', results)
    f.close()


# 主函数运行
if __name__ == '__main__':
    main()
