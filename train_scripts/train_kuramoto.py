import torch
from torch import nn
from Dataset import DataSet
from NN import NN
from Systems import Kuramoto
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
    parser.add_argument('--hidden_size_for', type=int, default=250, help = 'Hidden layer size for the forward transformation.')
    parser.add_argument('--epochs_for', type=int, default=10, help = 'Number of epochs for the forward transformation.')
    parser.add_argument('--lr_for', type=float, default=1e-4, help = 'Learning rate for the forward transformation.')
    parser.add_argument('--lambd', type=float, default=1e-3, help = 'Weight for PDE loss.')
    parser.add_argument('--batch_size_for', type=int, default=32, help = 'Batch size for the forward transformation.')
    parser.add_argument('--no_pde', action='store_true', default=False, help = 'No PDE training if True.')
    parser.add_argument('--dropout_for', type=int, default=0.0, help = 'Dropout probability for forward transformation.')
    parser.add_argument('--out_dir', type=str, default='./models', help = 'Output directory for the trained model.')

    parser.add_argument('--dropout_inv', type=int, default=0.0, help = 'Dropout probability for inverse transformation.')
    parser.add_argument('--num_hidden_inv', type=int, default=2, help = 'Number of hidden layers for the inverse transformation.')
    parser.add_argument('--hidden_size_inv', type=int, default=250, help = 'Hidden layer size for the inverse transformation.')
    parser.add_argument('--epochs_inv', type=int, default=10, help = 'Number of epochs for the inverse transformation.')
    parser.add_argument('--lr_inv', type=float, default=1e-4, help = 'Learning rate for the inverse transformation.')
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

    # --------------------- Kuramoto Oscillator setup --------------------- 
    print('Loading system settings.')
    coupling = 3

    A = np.array([[0.82609078, 0.766099  , 0.04770744, 0.95634223, 0.48535484, 0.82770635, 0.82568269, 0.41482854, 0.10805105, 0.84533148],
                    [0.16793279, 0.52216174, 0.32375506, 0.21255903, 0.74318917, 0.27801737, 0.62963795, 0.46563399, 0.87285444, 0.44818149],
                    [0.85739688, 0.35466085, 0.23081837, 0.77816442, 0.44434047, 0.04087587, 0.96132654, 0.20077639, 0.35555799, 0.5465003 ],
                    [0.66843566, 0.63740146, 0.92209938, 0.83202012, 0.11184414, 0.60635776, 0.70997797, 0.85543721, 0.97555328, 0.59325376],
                    [0.35215001, 0.01401137, 0.49319654, 0.17891016, 0.0477552 , 0.85102004, 0.95226833, 0.60082744, 0.80278956, 0.56294354],
                    [0.63712071, 0.12634065, 0.71471804, 0.4522626 , 0.82374189, 0.42226771, 0.71647475, 0.70623256, 0.66294877, 0.40769281],
                    [0.485451  , 0.7179554 , 0.00683135, 0.76084085, 0.71155059, 0.65390833, 0.55031198, 0.94644044, 0.84509827, 0.29303649],
                    [0.00997047, 0.7187184 , 0.84228867, 0.07802682, 0.2649064 , 0.72994764, 0.94594025, 0.50939704, 0.78263394, 0.3596041 ],
                    [0.95868957, 0.89330904, 0.36732745, 0.41224862, 0.23902724, 0.81451302, 0.06999864, 0.49742364, 0.40118135, 0.23091774],
                    [0.64234836, 0.89619788, 0.0335701 , 0.41857516, 0.19653272,0.63108802, 0.47687404, 0.89418281, 0.26760051, 0.58927806]])

    a = np.ones([21,1])
    b = np.eye(5)
    K = np.kron(a,b)
    a = np.diag(np.linspace(15, 20, 21))
    M = -np.kron(a,b)

    a = 0
    b = 30
    N = 4000
    limits = np.array([[-2, 2], [-2, 2], [-2, 2],[-2, 2], [-2, 2], [-2, 2],[-2, 2], [-2, 2], [-2, 2],[-2, 2]])    # Sample space for normal datapoints
    num_ic = 100       # Number of initial conditions to be sampled
    kuramoto = Kuramoto(10, coupling, adj_mat = A)

    print('Generating forward dataset.')
    trainset = DataSet(kuramoto, M, K, a, b, N, num_ic, limits, 8888, PINN_sample_mode = 'split traj', data_gen_mode = 'forward')
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
            'system':kuramoto,
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
    save_model(T_net, config, out_dir / f'kuramoto_T_forward_lr{learning_rate}_batch{batch_size}_epoch{epochs}')
    # -----------------------------------------------------

    # --------------------- Train Inverse --------------------

    print('Generating inverse dataset.')
    trainset = DataSet(kuramoto, M, K, a, b, N, num_ic, limits, 8888, PINN_sample_mode = 'no physics', data_gen_mode = 'negative forward', pretrained_T=T_net)
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
    save_model(T_inv_net, config, out_dir / f'kuramoto_T_inverse_lr{learning_rate}_batch{batch_size}_epoch{epochs}')
   # ------------------------------------------------------
   

if __name__ == '__main__':
    main()
