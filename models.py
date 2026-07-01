import torch
from torch import nn



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth, dropout=.2, smax=False):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(0, depth - 2):
            self.layers.append(nn.BatchNorm1d(hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.BatchNorm1d(hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        if smax:
            self.layers.append(nn.LogSoftmax(dim=-1))   #Beware of the log! 

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

            