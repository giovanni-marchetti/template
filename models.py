import torch
from torch import nn



class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, depth):
        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_dim)])
        for _ in range(0, depth - 2):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        
    def forward(self, x):
        for l in self.layers:  
            x = l(x)
        return x

            