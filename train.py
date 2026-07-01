import torch
from torchvision.datasets import MNIST
from torch.utils.tensorboard import SummaryWriter
import os
import argparse

from models import *
from losses import *
from datasets import *


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device: ', device)



"""
Parameters of the model
"""
parser = argparse.ArgumentParser()
parser.add_argument('--num_ep', type=int, default=100, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--hidden_dim', type=int, default=5, help='hidden dimension')
parser.add_argument('--seed', type=int, default=None, help='random seed')
args = parser.parse_args()


num_ep = args.num_ep
batch_size = args.batch_size
lr = args.lr
hidden_dim = args.hidden_dim
seed = args.seed 
if seed != None:
    torch.manual_seed(seed)       


print_interval = 10
save_interval = 10
save_dir = "results"
log_dir = os.path.join(save_dir, "tensorboard")




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

writer = SummaryWriter(log_dir=log_dir)



"""
Training loop
"""
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

            if batch_idx % print_interval == 0:
                print(f"Train epoch: {epoch}, Batch: {batch_idx} of {len(data_loader)} Loss: {loss:.3}")

        writer.add_scalar('error/train', tot_error / float(len(data_loader)), epoch)


    elif mode == 'test':
        model.eval()                
        for batch_idx, (x, y) in enumerate(data_loader):     

            x = x.to(device) 
            y = y.to(device)

            out = model(x)
            loss = test_loss_fn(out, y)
            
            tot_error += loss.item()

        print(f'Test epoch: {epoch}, loss: {loss:.3}')

        writer.add_scalar('error/test', tot_error / float(len(data_loader)), epoch)



if __name__ == "__main__":

    for i in range(1, num_ep + 1):
        print(f'Epoch {i}')

        train(i, train_loader, mode='train')
        with torch.no_grad():
            train(i, test_loader, mode='test')

        if i % save_interval == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, 'checkpoints', f'checkpoint_{i}.pth'))

    writer.close()