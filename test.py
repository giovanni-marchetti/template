import torch
import numpy as np
import os


from models import *
from losses import *
from datasets import * 


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)


idx = 10 
save_dir = "results"

dset = GaussianDataset(in_dim=10, out_dim=2)
x, _ = dset.__getitem__(0) 
x = torch.Tensor(x).unsqueeze(0)

model = MLP(in_dim=10, out_dim=2, hidden_dim=5, depth=3).to(device)
model.load_state_dict(torch.load(os.path.join(save_dir, 'checkpoints', f'checkpoint_{idx}.pth'))) 
model.eval()

with torch.no_grad():
    y = model(x).squeeze(0)
    print(y) 
