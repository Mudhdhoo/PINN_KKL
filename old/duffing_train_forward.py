import torch
from torch import nn
import Systems
import numpy as np
from NN import NN
from Dataset import DataSet
from Normalizer import Normalizer
import torch.nn.functional as F
import argparse
from pde_utils import pde_loss
from pathlib import Path
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hidden', type=int, default=3, help = 'Number of hidden layers.')
    parser.add_argument('--hidden_size', type=int, default=250, help = 'Hidden layer size.')
    parser.add_argument('--epochs', type=int, default=15, help = 'Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help = 'Learning rate.')
    parser.add_argument('--lambd', type=float, default=1e-2, help = 'Weight for PDE loss.')
    parser.add_argument('--batch_size', type=int, default=32, help = 'Batch size.')
    parser.add_argument('--no_pde', action='store_true', default=False, help = 'No PDE training if True.')
    parser.add_argument('--out_dir', type=str, default='./models', help = 'Output directory for the trained model.')

    return parser.parse_args()

def save_model(model,conf,filepath):
    save_info = {
        'model': model.state_dict(),
        'config':conf
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train(conf):
    train_dataloader = conf['train_dataloader']
    epochs = conf['epochs']
    optimizer = conf['optimizer']
    loss_fn = conf['loss_fn']
    no_pde = conf['no_pde']
    device = conf['device']
    M = conf['M_mat']
    K = conf['K_mat']
    system = conf['system']
    lambd = conf['lambda']
    time = conf['time'].to(device)
    T_net = conf['model'].to(device)
    T_net.mode = 'normal'

    train_log = []
    val_log = []
    for epoch in range(epochs):
        loss_sum = 0
        for idx, data in tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch}',disable = False):
            x, z, y, x_ph, y_ph = data      # Normal and physics data
            x, z, y = x.to(device), z.to(device), y.to(device)
            if not no_pde:
                x_ph, y_ph = x_ph.to(device), y_ph.to(device)

            z_hat = T_net(x)     # Forward pass
            loss_mse = loss_fn(z_hat, z)    # MSE loss
            if not no_pde:
                loss_pde = pde_loss(T_net, x, y, z_hat, time, system, M, K, device, reduction = 'mean')     # Pde loss
                loss = loss_mse + lambd*loss_pde
            else:
                loss = loss_mse

            loss_sum += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss = (loss_sum / idx).item()
        train_log.append(training_loss)
        print('Epoch:', epoch+1, 'Training loss:', training_loss)    # Average loss per epoch

def main():
    args = get_args()
    out_dir = Path(args.out_dir)

    # --------------------- Reverse Duffing Oscillator --------------------- 
    limits_normal = np.array([[-1, 1], [-1, 1]])    # Sample space for normal datapoints
    a = 0   # start
    b = 50  # end
    N = 1250          # Number of intervals for RK4
    num_ic = 60       # Number of initial conditions to be sampled

    M = np.diag([-1,-2,-3,-4,-5])    # Choose M matrix
    K = np.array([[1,1,1,1,1]]).T    # Choose K matrix

    revduff = Systems.RevDuff(5, add_noise=False)
    trainset = DataSet(revduff, M, K, a, b, N, num_ic, limits_normal, 8888, PINN_sample_mode = 'split traj', data_gen_mode = 'forward')
    # ------------------------------------------------------------------------ 

    # --------------------- Training Setup ---------------------
    x_size = trainset.system.x_size
    z_size = trainset.system.z_size
    num_hidden = args.num_hidden
    hidden_size = args.hidden_size
    activation = F.relu
    normalizer = Normalizer(trainset)
    T_net = NN(num_hidden, hidden_size, x_size, z_size, activation, normalizer)    

    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.batch_size
    optimizer = torch.optim.Adam(T_net.parameters(), lr = learning_rate)
    loss_fn = nn.MSELoss(reduction = 'mean')
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
    conf = {'train_dataloader':train_dataloader,
            'epochs':epochs,
            'optimizer':optimizer,
            'model':T_net,
            'loss_fn':loss_fn,
            'no_pde':args.no_pde,
            'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'M_mat':M,
            'K_mat':K,
            'system':revduff,
            'time':trainset.time,
            'lambda':args.lambd}
    # -----------------------------------------------------------

    # --------------------- Training ---------------------
    train(conf)
    config = {'num_hidden':num_hidden,
              'hidden_size':hidden_size,
              'x_size':x_size,
              'z_size':z_size,
              'normalizer':normalizer,
              'activation':activation}

    # Save the model
    save_model(T_net, config, out_dir / f'T_forward_lr{learning_rate}_batch{batch_size}_epoch{epochs}')
    # -----------------------------------------------------
   


if __name__ == '__main__':
    main()