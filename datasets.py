import torch
from torch.utils.data import Dataset


class GaussianDataset(Dataset):

    def __init__(self, num_samples=1000, in_dim=10, out_dim=1, correlated=True):
        super().__init__()
        
        self.x = torch.randn(num_samples, in_dim)
        if correlated: 
            A = torch.randn(1, out_dim, in_dim)
            self.y = (A @ self.x.unsqueeze(-1)).squeeze(-1)      
        else: 
            self.y = torch.randn(num_samples, out_dim)
        
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


