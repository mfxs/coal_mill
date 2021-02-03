# Import Packages
import math
import torch
import cvxpy as cp
import torch.nn as nn
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from sklearn.gaussian_process.kernels import RBF
from lssvr_in_Python_master.lssvrFunctions import *
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# Calculate R2和RMSE
def r2_rmse(y_true, y_pred, name, file):
    r2 = r2_score(y_true, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))
    for i in range(r2.shape[0]):
        print('{}: R2: {:.2f} RMSE: {:.4f}'.format(name[i], r2[i], rmse[i]))
        file.write('{}: R2: {:.2f} RMSE: {:.4f}\n'.format(name[i], r2[i], rmse[i]))
    print('Averaged R2: {:.2f}'.format(np.mean(r2)))
    file.write('Averaged R2: {:.2f}\n'.format(np.mean(r2)))
    print('Averaged RMSE: {:.4f}\n'.format(np.mean(rmse)))
    file.write('Averaged RMSE: {:.4f}\n\n'.format(np.mean(rmse)))
    return r2, rmse


# Calculate Adjacency Matrix
def adjacency_matrix(x, mode, gpu=torch.device('cuda:0'), graph_reg=0.05, self_con=0.2, scale=0.4, epsilon=0.1):
    # RBF kernel function
    if mode == 'rbf':
        kernel = RBF(length_scale=scale)
        A = kernel(x, x)

    # Pearson Correlation Coefficient
    elif mode == 'pearson':
        A = np.corrcoef(x.T)

    # Sparse Coding
    elif mode == 'sc':
        A = cp.Variable((x.shape[1], x.shape[1]))
        term1 = cp.norm(x * A - x, p='fro')
        term2 = cp.norm1(A)
        constraints = []
        for i in range(x.shape[1]):
            constraints.append(A[i, i] == 0)
            for j in range(x.shape[1]):
                constraints.append(A[i, j] >= 0)
        constraints.append(A == A.T)
        objective = cp.Minimize(term1 + graph_reg * term2)
        prob = cp.Problem(objective, constraints)
        prob.solve()
        A = A.value
        A = A + self_con * np.eye(x.shape[1])

    # Omit small values
    A[np.abs(A) < epsilon] = 0

    # Normalization
    D = np.diag(np.sum(A, axis=1) ** (-0.5))
    A = np.matmul(np.matmul(D, A), D)
    A = torch.tensor(A, dtype=torch.float32, device=gpu)

    return A


# LSSVR
class LssvrModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, c=100, sigma=0.1):
        super(LssvrModel, self).__init__()

        # Parameter Assignment
        self.c = c
        self.sigma = sigma
        self.alpha = []
        self.bias = []

        # Initializa Scaler
        self.scaler = StandardScaler()

    # Train
    def fit(self, X, y):
        self.num_y = y.shape[1]
        self.X = self.scaler.fit_transform(X)
        for i in range(self.num_y):
            alpha, bias = getParams(self.X, np.array(y)[:, i].reshape(-1, 1), 1 / self.sigma ** 2, self.c)
            self.alpha.append(alpha)
            self.bias.append(bias)

        return self

    # Test
    def predict(self, X):
        X = self.scaler.transform(X)
        y = np.zeros((X.shape[0], self.num_y))
        for i in range(self.num_y):
            y[:, i] = compute_lssvr(self.X, X, 1 / self.sigma ** 2, self.alpha[i], self.bias[i]).squeeze()

        return y


# MyDataset
class MyDataset(Dataset):

    def __init__(self, data, label, mode='2D'):
        self.data, self.label, self.mode = data, label, mode

    def __getitem__(self, index):
        if self.mode == '2D':
            return self.data[index, :], self.label[index, :]
        elif self.mode == '3D':
            return self.data[:, index, :], self.label[:, index, :]

    def __len__(self):
        if self.mode == '2D':
            return self.data.shape[0]
        elif self.mode == '3D':
            return self.data.shape[1]


# Graph Convolutional Operation
class GraphConvolution(nn.Module):

    # Initialization
    def __init__(self, n_input, n_output):
        super(GraphConvolution, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.weight = Parameter(torch.FloatTensor(n_input, n_output))
        self.reset_parameters()

    # Forward Propagation
    def forward(self, x, adj):
        res = torch.matmul(adj, torch.matmul(x, self.weight))
        return res

    # Weight Reset
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)


# FCN
class FCN(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, hidden_layers=(256,)):
        super(FCN, self).__init__()

        # Parameter Assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.hidden_layers = hidden_layers
        self.network_structure = [dim_X, ] + list(hidden_layers) + [dim_y, ]

        # Model Creation
        self.net = nn.ModuleList()
        for i in range(len(hidden_layers)):
            self.net.append(
                nn.Sequential(nn.Linear(self.network_structure[i], self.network_structure[i + 1]), nn.ReLU()))
        self.net.append(nn.Linear(self.network_structure[-2], self.network_structure[-1]))

    # Forward Propagation
    def forward(self, X):
        feat = X
        for i in self.net:
            feat = i(feat)

        return feat


# FCN
class FcnModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, hidden_layers=(256,), n_epoch=200, batch_size=64, lr=0.001, weight_decay=0.01,
                 step_size=50, gamma=0.5):
        super(FcnModel, self).__init__()

        # Parameter Assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.hidden_layers = hidden_layers
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        # Initialize Scaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Model Creation
        self.loss_hist = []
        self.gpu = torch.device('cuda:0')
        self.model = FCN(dim_X, dim_y, hidden_layers).to(self.gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.gpu),
                            torch.tensor(y, dtype=torch.float32, device=self.gpu))
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # Test
    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu)
        self.model.eval()
        y = self.scaler_y.inverse_transform(self.model(X).detach().cpu().numpy())

        return y


# LSTM
class LSTM(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,), fc=(256,)):
        super(LSTM, self).__init__()

        # # Parameter Assignment
        # self.dim_X = dim_X
        # self.dim_y = dim_y
        # self.n_lstm = [dim_X, ] + list(lstm)
        # self.n_fc = [lstm[-1], ] + list(fc) + [1, ]
        #
        # # Model Creation
        # self.lstm = nn.ModuleList()
        # self.fc = nn.ModuleList()
        # for i in range(dim_y):
        #     self.lstm.append(nn.ModuleList())
        #     self.fc.append(nn.ModuleList())
        #     for j in range(len(lstm)):
        #         self.lstm[-1].append(nn.LSTM(self.n_lstm[j], self.n_lstm[j + 1]))
        #     for j in range(len(fc)):
        #         self.fc[-1].append(nn.Sequential(nn.Linear(self.n_fc[j], self.n_fc[j + 1]), nn.ReLU()))
        #     self.fc[-1].append(nn.Linear(self.n_fc[-2], self.n_fc[-1]))

        # Parameter Assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.n_lstm = [dim_X, ] + list(lstm)
        self.n_fc = [lstm[-1], ] + list(fc) + [dim_y, ]

        # Model Creation
        self.lstm = nn.ModuleList()
        for i in range(len(lstm)):
            self.lstm.append(nn.LSTM(self.n_lstm[i], self.n_lstm[i + 1]))
        self.fc = nn.ModuleList()
        for i in range(len(fc)):
            self.fc.append(nn.Sequential(nn.Linear(self.n_fc[i], self.n_fc[i + 1]), nn.ReLU()))
        self.fc.append(nn.Linear(self.n_fc[-2], self.n_fc[-1]))

    # Forward Propagation
    def forward(self, X):
        # feat_list = []
        #
        # for i in range(self.dim_y):
        #     feat = X
        #
        #     for j in self.lstm[i]:
        #         feat = j(feat)[0]
        #
        #     for j in self.fc[i]:
        #         feat = j(feat)
        #
        #     feat_list.append(feat)
        #
        # res = torch.stack(feat_list, 3).squeeze()
        #
        # return res

        feat = X

        for i in self.lstm:
            feat = i(feat)[0]

        for i in self.fc:
            feat = i(feat)

        feat.squeeze_()

        return feat


# LSTM
class LstmModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, lstm=(1024,), fc=(256,), seq_len=30, n_epoch=200, batch_size=64, lr=0.001,
                 weight_decay=0.01, step_size=50, gamma=0.5):
        super(LstmModel, self).__init__()

        # Parameter Assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.lstm = lstm
        self.fc = fc
        self.seq_len = seq_len
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        # Initialize Scaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Model Creation
        self.loss_hist = []
        self.gpu = torch.device('cuda:0')
        self.model = LSTM(dim_X, dim_y, lstm, fc).to(self.gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_len + 1):
            X_3d.append(X[i:i + self.seq_len, :])
            y_3d.append(y[i:i + self.seq_len, :])
        X_3d = np.stack(X_3d, 1)
        y_3d = np.stack(y_3d, 1)
        dataset = MyDataset(torch.tensor(X_3d, dtype=torch.float32, device=self.gpu),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.gpu), '3D')
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(1, 0, 2)
                batch_y = batch_y.permute(1, 0, 2)
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # Test
    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu).unsqueeze(1)
        self.model.eval()
        y = self.scaler_y.inverse_transform(self.model(X).detach().cpu().numpy())

        return y


# GCN
class GCN(nn.Module):

    # Initialization
    def __init__(self, dim_X, dim_y, gc=(1024,), fc=(256,)):
        super(GCN, self).__init__()

        # # Parameter Assignment
        # self.dim_X = dim_X
        # self.dim_y = dim_y
        # self.n_gc = [dim_X, ] + list(gc)
        # self.n_fc = [gc[-1], ] + list(fc) + [1, ]
        #
        # # Model Creation
        # self.gc = nn.ModuleList()
        # self.fc = nn.ModuleList()
        # self.act = nn.ReLU()
        # for i in range(self.dim_y):
        #     self.gc.append(nn.ModuleList())
        #     self.fc.append(nn.ModuleList())
        #     for j in range(len(gc)):
        #         self.gc[-1].append(GraphConvolution(self.n_gc[j], self.n_gc[j + 1]))
        #     for j in range(len(fc)):
        #         self.fc[-1].append(nn.Sequential(nn.Linear(self.n_fc[j], self.n_fc[j + 1]), nn.ReLU()))
        #     self.fc[-1].append(nn.Linear(self.n_fc[-2], self.n_fc[-1]))

        # Parameter Assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.n_gc = [dim_X, ] + list(gc)
        self.n_fc = [gc[-1], ] + list(fc) + [dim_y, ]

        # Model Creation
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.n_gc[i], self.n_gc[i + 1]))
        self.fc = nn.ModuleList()
        for i in range(len(fc)):
            self.fc.append(nn.Sequential(nn.Linear(self.n_fc[i], self.n_fc[i + 1]), nn.ReLU()))
        self.fc.append(nn.Linear(self.n_fc[-2], self.n_fc[-1]))
        self.act = nn.ReLU()

    # Forward Propagation
    def forward(self, x, gpu=torch.device('cuda:0'), graph_reg=0.05, self_con=0.2):
        adj = adjacency_matrix(x.cpu().numpy(), 'rbf', gpu, graph_reg, self_con)
        # feat_list = []
        #
        # for i in range(self.dim_y):
        #     feat = x
        #
        #     for j in self.gc[i]:
        #         feat = self.act(j(feat, adj))
        #
        #     for j in self.fc[i]:
        #         feat = j(feat)
        #
        #     feat_list.append(feat)
        #
        # res = torch.stack(feat_list, 2).squeeze()
        #
        # return res

        feat = x

        for i in self.gc:
            feat = self.act(i(feat, adj))

        for i in self.fc:
            feat = i(feat)

        return feat


# GCN
class GcnModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, dim_X, dim_y, gc=(1024), fc=(256,), graph_reg=0.05, self_con=0.2, n_epoch=200, batch_size=64,
                 lr=0.001, weight_decay=0.01, step_size=50, gamma=0.5):
        super(GcnModel, self).__init__()

        # Parameter Assignment
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.gc = gc
        self.fc = fc
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        # Initialize Scaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Model Creation
        self.loss_hist = []
        self.gpu = torch.device('cuda:0')
        self.model = GCN(dim_X, dim_y, gc, fc).to(self.gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = torch.tensor(self.scaler_X.fit_transform(X), dtype=torch.float32, device=self.gpu)
        y = torch.tensor(self.scaler_y.fit_transform(y), dtype=torch.float32, device=self.gpu)
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            self.optimizer.zero_grad()
            output = self.model(X, self.gpu, self.graph_reg, self.self_con)
            loss = self.criterion(output, y)
            self.loss_hist[-1] = loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # Test
    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu)
        self.model.eval()
        y = self.scaler_y.inverse_transform(self.model(X).detach().cpu().numpy())

        return y


# MC-GCN
class MCGCN(nn.Module):

    # Initialization
    def __init__(self, n_input, in_fc, gc, out_fc, n_output):
        super(MCGCN, self).__init__()

        # Parameter Assignment
        self.n_input = n_input
        self.net_in_fc = [n_input, ] + list(in_fc)
        self.net_gc = [in_fc[-1], ] + list(gc)
        self.net_out_fc = [gc[-1], ] + list(out_fc) + [1, ]
        self.n_output = n_output
        self.act = nn.ReLU()

        # Input FC
        self.in_fc = nn.ModuleList()
        for i in range(len(in_fc)):
            temp = nn.ModuleList()
            for j in range(n_output):
                temp.append(nn.Sequential(nn.Linear(self.net_in_fc[i], self.net_in_fc[i + 1]), nn.ReLU()))
            self.in_fc.append(temp)

        # GC
        self.gc = nn.ModuleList()
        for i in range(len(gc)):
            self.gc.append(GraphConvolution(self.net_gc[i], self.net_gc[i + 1]))

        # Output FC
        self.out_fc = nn.ModuleList()
        for i in range(len(out_fc)):
            temp = nn.ModuleList()
            for j in range(n_output):
                temp.append(nn.Sequential(nn.Linear(self.net_out_fc[i], self.net_out_fc[i + 1]), nn.ReLU()))
            self.out_fc.append(temp)
        temp = nn.ModuleList()
        for j in range(n_output):
            temp.append(nn.Linear(self.net_out_fc[-2], self.net_out_fc[-1]))
        self.out_fc.append(temp)

    # Forward Propagation
    def forward(self, x, adj):
        feat_list = []
        res_list = []

        # Input FC
        for i in range(self.n_output):
            feat = x
            for fc in self.in_fc:
                feat = fc[i](feat)
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=1)

        # GC
        for gc in self.gc:
            feat = gc(feat, adj)
            feat = self.act(feat)

        # Output FC
        for i in range(self.n_output):
            res = feat[:, i, :]
            for fc in self.out_fc:
                res = fc[i](res)
            res_list.append(res.squeeze())
        res = torch.stack(res_list, dim=1)

        return res


# MC-GCN
class McgcnModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, n_input, in_fc, gc, out_fc, n_output, graph_reg=0.05, self_con=0.2, n_epoch=200, batch_size=64,
                 lr=0.001, weight_decay=0.1, step_size=50, gamma=0.5):
        super(McgcnModel, self).__init__()

        # Parameter Assignment
        self.n_input = n_input
        self.in_fc = in_fc
        self.gc = gc
        self.out_fc = out_fc
        self.n_output = n_output
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        # Initialize Scaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Model Creation
        self.loss_hist = []
        self.gpu = torch.device('cuda:0')
        self.model = MCGCN(n_input, in_fc, gc, out_fc, n_output).to(self.gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        self.adj = adjacency_matrix(y, 'sc', self.gpu, self.graph_reg, self.self_con)
        dataset = MyDataset(torch.tensor(X, dtype=torch.float32, device=self.gpu),
                            torch.tensor(y, dtype=torch.float32, device=self.gpu))
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                self.optimizer.zero_grad()
                output = self.model(batch_X, self.adj)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # Test
    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu)
        self.model.eval()
        y = self.scaler_y.inverse_transform(self.model(X, self.adj).detach().cpu().numpy())

        return y


# GC-LSTM
class GCLSTM(nn.Module):

    # Initialization
    def __init__(self, n_input, n_lstm, n_gc, n_fc, n_output):
        super(GCLSTM, self).__init__()

        # Parameter Assignment
        self.n_input = n_input
        self.n_lstm = [n_input, ] + list(n_lstm)
        self.n_gc = [n_lstm[-1], ] + list(n_gc)
        self.n_fc = [n_gc[-1], ] + list(n_fc) + [1, ]
        self.n_output = n_output
        self.lstm = nn.ModuleList()
        self.gc = nn.ModuleList()
        self.fc = nn.ModuleList()
        self.act = nn.ReLU()

        # GC
        for i in range(len(n_gc)):
            self.gc.append(GraphConvolution(self.n_gc[i], self.n_gc[i + 1]))

        # Each Channel
        for i in range(n_output):

            # LSTM
            self.lstm.append(nn.ModuleList())
            for j in range(len(n_lstm)):
                self.lstm[-1].append(nn.LSTM(self.n_lstm[j], self.n_lstm[j + 1]))

            # FC
            self.fc.append(nn.ModuleList())
            for j in range(len(n_fc)):
                self.fc[-1].append(nn.Sequential(nn.Linear(self.n_fc[j], self.n_fc[j + 1]), nn.ReLU()))
            self.fc[-1].append(nn.Linear(self.n_fc[-2], self.n_fc[-1]))

    # Forward Propagation
    def forward(self, x, adj):
        feat_list = []
        res_list = []

        # LSTM
        for i in range(self.n_output):
            feat = x
            for j in self.lstm[i]:
                feat = j(feat)[0]
            feat_list.append(feat)
        feat = torch.stack(feat_list, dim=2)

        # GC
        for i in self.gc:
            feat = i(feat, adj)
            feat = self.act(feat)

        # FC
        for i in range(self.n_output):
            res = feat[:, :, i, :]
            for j in self.fc[i]:
                res = j(res)
            res_list.append(res)
        res = torch.stack(res_list, 3).squeeze()

        return res


# GC-LSTM
class GclstmModel(BaseEstimator, RegressorMixin):

    # Initialization
    def __init__(self, n_input, n_lstm, n_gc, n_fc, n_output, seq_len=30, graph_reg=0.05, self_con=0.2, n_epoch=200,
                 batch_size=64, lr=0.001, weight_decay=0.1, step_size=50, gamma=0.5):
        super(GclstmModel, self).__init__()

        # Parameter Assignment
        self.n_input = n_input
        self.n_lstm = n_lstm
        self.n_gc = n_gc
        self.n_fc = n_fc
        self.n_output = n_output
        self.seq_len = seq_len
        self.graph_reg = graph_reg
        self.self_con = self_con
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma

        # Initialize Scaler
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        # Model Creation
        self.loss_hist = []
        self.gpu = torch.device('cuda:0')
        self.model = GCLSTM(n_input, n_lstm, n_gc, n_fc, n_output).to(self.gpu)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        self.criterion = nn.MSELoss(reduction='sum')

    # Train
    def fit(self, X, y):
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        self.adj = adjacency_matrix(y, 'sc', self.gpu, self.graph_reg, self.self_con)
        X_3d = []
        y_3d = []
        for i in range(X.shape[0] - self.seq_len + 1):
            X_3d.append(X[i:i + self.seq_len, :])
            y_3d.append(y[i:i + self.seq_len, :])
        X_3d = np.stack(X_3d, 1)
        y_3d = np.stack(y_3d, 1)
        dataset = MyDataset(torch.tensor(X_3d, dtype=torch.float32, device=self.gpu),
                            torch.tensor(y_3d, dtype=torch.float32, device=self.gpu), '3D')
        self.model.train()
        for i in range(self.n_epoch):
            self.loss_hist.append(0)
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.permute(1, 0, 2)
                batch_y = batch_y.permute(1, 0, 2)
                self.optimizer.zero_grad()
                output = self.model(batch_X, self.adj)
                loss = self.criterion(output, batch_y)
                self.loss_hist[-1] += loss.item()
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            print('Epoch: {}, Loss: {}'.format(i + 1, self.loss_hist[-1]))
        print('Optimization finished!')

        return self

    # Test
    def predict(self, X):
        X = torch.tensor(self.scaler_X.transform(X), dtype=torch.float32, device=self.gpu).unsqueeze(1)
        self.model.eval()
        y = self.scaler_y.inverse_transform(self.model(X, self.adj).detach().cpu().numpy())

        return y
