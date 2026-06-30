import torch
from torchvision.datasets import MNIST
import numpy as np
import os
import argparse

from models import *
from losses import *
from datasets import * 


# # torch.manual_seed(42)       
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)
# device = 'cpu'    




"""
Parameters of the model
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_ep', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_dim', type=int, default=5, help='hidden dimension')
args = parser.parse_args()

num_ep = args.num_ep
batch_size = args.batch_size
lr = args.lr
hidden_dim = args.hidden_dim


log_interval = 10
save_interval = 10
save_dir = "results"




"""
Initialize dataset
"""
# dset_train = MNIST(root='./', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# dset_test = MNIST(root='./', train=False, transform=torchvision.transforms.ToTensor(), download=True)
dset = GaussianDataset(in_dim=10, out_dim=2)
dset_train, dset_test = torch.utils.data.random_split(dset, [0.8, 0.2])
train_loader = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dset_test, batch_size=batch_size, shuffle=False)




"""
Initialize model and optimizer
"""
model = MLP(in_dim=10, out_dim=2, hidden_dim=hidden_dim, depth=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loss_fn = LpLoss
test_loss_fn = LpLoss 



"""
Training loop
"""
train_errors = [] 
test_errors = []

def train(epoch, data_loader, mode='train'):
    tot_error = 0.  

    if mode == 'train':
        model.train()
        for batch_idx, (x, y) in enumerate(data_loader):
            
            optimizer.zero_grad()

            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = train_loss_fn(out, y)

            loss.backward()
            optimizer.step()

            tot_error += loss.item()

            if batch_idx % log_interval == 0:
                print(f"Train epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3}")

        train_errors.append(tot_error / float(len(data_loader))) 
        np.save(os.path.join(save_dir, 'train_errors.npy'), np.array(train_errors))


    elif mode == 'test':
        model.eval()                
        for batch_idx, (x, y) in enumerate(data_loader):     

            x = x.to(device) 
            y = y.to(device)

            out = model(x)
            loss = test_loss_fn(out, y)
            
            tot_error += loss.item()

        print(f'Test epoch: {epoch}, loss: {loss:.3}')

        test_errors.append(tot_error / float(len(data_loader))) 
        np.save(os.path.join(save_dir, 'test_errors.npy'), np.array(test_errors))



if __name__ == "__main__":

    for i in range(1, num_ep + 1):
        print(f'Epoch {i}')

        train(i, train_loader, mode='train')
        with torch.no_grad():
            train(i, test_loader, mode='test')

        if i % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoints', f'checkpoint_{i}.pth'))