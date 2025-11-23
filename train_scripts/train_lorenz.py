import torch
from torch import nn
from Dataset import DataSet
from NN import NN
from Systems import Lorenz
import numpy as np
from Normalizer import Normalizer
import torch.nn.functional as F
import argparse
from pde_utils import pde_loss
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hidden_for', type=int, default=2, help = 'Number of hidden layers for the forward transformation.')
    parser.add_argument('--hidden_size_for', type=int, default=350, help = 'Hidden layer size for the forward transformation.')
    parser.add_argument('--epochs_for', type=int, default=15, help = 'Number of epochs for the forward transformation.')
    parser.add_argument('--lr_for', type=float, default=1e-3, help = 'Learning rate for the forward transformation.')
    parser.add_argument('--lambd', type=float, default=1e-2, help = 'Weight for PDE loss.')
    parser.add_argument('--batch_size_for', type=int, default=32, help = 'Batch size for the forward transformation.')
    parser.add_argument('--no_pde', action='store_true', default=False, help = 'No PDE training if True.')
    parser.add_argument('--dropout_for', type=int, default=0.0, help = 'Dropout probability for forward transformation.')
    parser.add_argument('--out_dir', type=str, default='./models', help = 'Output directory for the trained model.')

    parser.add_argument('--dropout_inv', type=int, default=0.0, help = 'Dropout probability for inverse transformation.')
    parser.add_argument('--num_hidden_inv', type=int, default=2, help = 'Number of hidden layers for the inverse transformation.')
    parser.add_argument('--hidden_size_inv', type=int, default=350, help = 'Hidden layer size for the inverse transformation.')
    parser.add_argument('--epochs_inv', type=int, default=15, help = 'Number of epochs for the inverse transformation.')
    parser.add_argument('--lr_inv', type=float, default=1e-3, help = 'Learning rate for the inverse transformation.')
    parser.add_argument('--batch_size_inv', type=int, default=32, help = 'Batch size for the inverse transformation.')

    return parser.parse_args()

def save_model(model,conf,filepath):
    save_info = {
        'model': model.state_dict(),
        'config':conf
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def train_forward(conf):
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
    scheduler = conf['scheduler']
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
        scheduler.step(training_loss)
        lr = scheduler.get_last_lr()
        print('Epoch:', epoch+1, 'lr:', lr, 'Training loss:', training_loss)    # Average loss per epoch

def train_inverse(conf):
    train_dataloader = conf['train_dataloader']
    epochs = conf['epochs']
    optimizer = conf['optimizer']
    loss_fn = conf['loss_fn']
    device = conf['device']
    scheduler = conf['scheduler']
    T_inv_net = conf['model'].to(device)

    T_inv_net.mode = 'normal'

    train_log = []
    for epoch in range(epochs):
        loss_sum = 0
        for idx, data in tqdm(enumerate(train_dataloader), desc=f'Epoch {epoch}'):
            x, z, y, _, _ = data      # Normal and physics data
            x, z, y = x.to(device), z.to(device), y.to(device)

            x_hat = T_inv_net(z)     # Forward pass
            loss = loss_fn(x_hat, x)    # MSE loss
            loss_sum += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        training_loss = (loss_sum / idx).item()
        train_log.append(training_loss)
        scheduler.step(training_loss)
        lr = scheduler.get_last_lr()
        print('Epoch:', epoch+1, 'lr:', lr, 'Training loss:', training_loss)    # Average loss per epoch

def main():
    args = get_args()
    out_dir = Path(args.out_dir)

    # --------------------- Reverse Duffing Oscillator setup --------------------- 
    print('Loading system settings.')
    rho = 28
    sigma = 10
    beta = 8/3

    a = 0
    b = 50
    N = 1000
    num_ic = 100

    M = np.array([[-2.12116939,  2.73877907, -0.75338041,  3.13947511,  1.00581224,
        -5.34548426, -5.34544528],
    [-5.91811463, -1.99160105, -0.26953991,  0.45654273, -0.56750459,
        -3.79023002, -3.00480715],
    [-1.27814843, -2.08375366, -2.60781906, -4.11186517,  0.33087248,
        0.08191241, -2.81998688],
    [ 0.69080313,  2.00687664,  3.02478991, -3.53231915, -0.89704804,
        -5.19153027,  3.16796666],
    [-1.16566857, -0.53979687, -0.30300461, -3.79753229, -2.7939018 ,
        -0.30176078, -2.61480264],
    [ 2.97899741,  3.97265572, -2.04951338, -1.08716448, -0.98741222,
        -4.7047881 , -1.38218644],
    [ 0.95887295,  0.43056828, -1.39564903, -3.18608794,  3.74388666,
        -3.88509855, -3.07468653]])

    K = np.ones([7,1])

    limits = np.array([[-1,1], [-1,1], [-1,1]])    # Sample space
    lorenz = Lorenz(rho, sigma, beta)
    print('Generating forward dataset.')
    trainset = DataSet(lorenz, M, K, a, b, N, num_ic, limits, 8888, PINN_sample_mode = 'split traj', data_gen_mode = 'forward')
    print(f'Dataset created with {len(trainset.x_data)} datapoints.')
    # ------------------------------------------------------------------------ 

    # --------------------- Training Setup ---------------------
    print('Loading training setup.')
    x_size = trainset.system.x_size
    z_size = trainset.system.z_size
    num_hidden = args.num_hidden_for
    hidden_size = args.hidden_size_for
    activation = F.relu
    normalizer = Normalizer(trainset)
    T_net = NN(num_hidden, hidden_size, x_size, z_size, activation, normalizer=normalizer, dropout_prob=args.dropout_for)    

    epochs = args.epochs_for
    learning_rate = args.lr_for
    batch_size = args.batch_size_for
    optimizer = torch.optim.Adam(T_net.parameters(), lr = learning_rate)
    loss_fn = nn.MSELoss(reduction = 'mean')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=1e-3)
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
            'system':lorenz,
            'time':trainset.time,
            'lambda':args.lambd,
            'scheduler':scheduler}
    # -----------------------------------------------------------

    # --------------------- Training ---------------------
    print('Begin training forward map.')
    train_forward(conf)
    config = {'num_hidden':num_hidden,
              'hidden_size':hidden_size,
              'x_size':x_size,
              'z_size':z_size,
              'normalizer':normalizer,
              'activation':activation}

    # Save the model
    save_model(T_net, config, out_dir / f'lorenz_T_forward_lr{learning_rate}_batch{batch_size}_epoch{epochs}')
    # -----------------------------------------------------

    # --------------------- Train Inverse --------------------

    print('Generating inverse dataset.')
    trainset = DataSet(lorenz, M, K, a, b, N, num_ic, limits, 8888, PINN_sample_mode = 'no physics', data_gen_mode = 'negative forward', pretrained_T=T_net)
    print(f'Dataset created with {len(trainset.x_data)} datapoints.')
    # ------------------------------------------------------------------------ 

    # --------------------- Training Setup ---------------------
    print('Loading training setup.')
    num_hidden = args.num_hidden_inv
    hidden_size = args.hidden_size_inv
    activation = F.relu
    normalizer = Normalizer(trainset)
    T_inv_net = NN(num_hidden, hidden_size, z_size, x_size, activation, normalizer=normalizer, dropout_prob=args.dropout_inv)    

    epochs = args.epochs_inv
    learning_rate = args.lr_inv
    batch_size = args.batch_size_inv
    optimizer = torch.optim.Adam(T_inv_net.parameters(), lr = learning_rate)
    loss_fn = nn.MSELoss(reduction = 'mean')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=1e-3)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True)
    conf = {'train_dataloader':train_dataloader,
            'epochs':epochs,
            'optimizer':optimizer,
            'model':T_inv_net,
            'loss_fn':loss_fn,
            'scheduler':scheduler,
            'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            }   
    # --------------------------------------------------------------

    # --------------------- Training ---------------------
    print('Begin training inverse map.')
    train_inverse(conf)
    config = {'num_hidden':num_hidden,
            'hidden_size':hidden_size,
            'x_size':x_size,
            'z_size':z_size,
            'normalizer':normalizer,
            'activation':activation}
    
    # Save the model
    save_model(T_inv_net, config, out_dir / f'lorenz_T_inverse_lr{learning_rate}_batch{batch_size}_epoch{epochs}')
   # ------------------------------------------------------
   

if __name__ == '__main__':
    main()