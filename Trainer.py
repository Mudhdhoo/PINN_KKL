from importlib.metadata import requires
import torch
import Loss_Calculator as L
from Loss_functions import *
from typing import TYPE_CHECKING, Optional, Callable
from Dataset import DataSet
from NN import Main_Network
from pde_utils import pde_loss

class Trainer:
    """
    Trainer class containing the training loop.

    ---------------- Parameters ----------------
    trainset: Dataset
        Dataset object containing training data.

    valset: Dataset
        Dataset object containing validation data.

    epochs: int
        Number of training epochs.

    optimizer: torch.optim
        Torch optimizer object.

    net: Main_Network
        Neural network object from NN.py.

    loss_fn: torch.nn
        Loss function from the torch.nn module.

    batch_size: int
        Dataloader batch size.

    shuffle: bool
        If True, the data is shuffled, otherwise not.

    scheduler: Optional
        Learning rate scheduler from torch.optim.lr_scheduler.

    reduction: str
        Either mean or sum.
        If mean, the mean loss of each batch is taken.
        If sum, the sum of the loss of each batch is used instead.
    """
    def __init__(self, trainset: DataSet, valset: DataSet, epochs: int, optimizer: torch.optim, net: Main_Network, loss_fn: torch.nn, batch_size: int, scheduler, shuffle: bool = True, reduction: str = 'mean') -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = trainset
        self.trainset = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = shuffle)
        self.valset = torch.utils.data.DataLoader(valset, batch_size = batch_size, shuffle = shuffle)
        self.epochs = epochs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.net = net.to(self.device)
        self.loss_calculator = L.Loss_Calculator(loss_fn, self.net, self.dataset, self.device)
        self.normalizer = net.normalizer
        self.reduction = reduction
        self.pde = PdeLoss_xz(self.dataset.M, self.dataset.K, self.dataset.system, self.loss_calculator, self.reduction)
        self.MSE = MSELoss(self.loss_calculator)
        print('Device:', self.device)

    def forward_pass(self, x, y, z, x_ph, y_ph, with_pde):
        self.net.mode = 'normal'
        time = self.dataset.time
        z_hat, x_hat, norm_z_hat, norm_x_hat = self.net(x)

        if self.normalizer != None:
            label_x = self.normalizer.Normalize(x, mode = 'normal').float()
            label_z = self.normalizer.Normalize(z, mode = 'normal').float()
        else:
            label_x = x
            label_z = z

        # Compute MSE loss    
        loss_normal = self.MSE(norm_x_hat, norm_z_hat, label_x, label_z)

        # Compute physics loss
        if with_pde:
            self.net.mode = 'physics'
            z_hat_ph = self.net(x_ph)[0]
            loss_pde = self.pde(x_ph, y_ph, z_hat_ph, time)
            loss = loss_normal + loss_pde
        else:
            loss = loss_normal

        return loss #, loss_normal, loss_pde
        
    def train(self, with_pde: bool = True) -> list:
        """
        Training loop.
        """
        train_log = []
        val_log = []
        for epoch in range(self.epochs):
            loss_sum = 0
            loss_sum_val = 0
            for idx, data in enumerate(self.trainset):
                x, z, y, x_ph, y_ph = data      # Normal and physics data
                x, z, y = x.to(self.device), z.to(self.device), y.to(self.device)
                if with_pde:
                    x_ph, y_ph = x_ph.to(self.device), y_ph.to(self.device)
                loss = self.forward_pass(x, y, z, x_ph, y_ph, with_pde)
                loss_sum += loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            training_loss = (loss_sum / idx).item()

            # ----------- Validation ----------

            for idx, data in enumerate(self.valset):
                x_val, z_val, y_val, x_ph_val, y_ph_val = data      # Normal and physics data
                x_val, z_val, y_val = x_val.to(self.device), z_val.to(self.device), y_val.to(self.device)
                if with_pde:
                    x_ph_val, y_ph_val = x_ph_val.to(self.device), y_ph_val.to(self.device)

                #self.net.mode = 'normal'
                with torch.no_grad():
                    loss_val = self.forward_pass(x_val, y_val, z_val, x_ph_val, y_ph_val, with_pde)
                loss_sum_val += loss_val

            validation_loss = (loss_sum_val / idx).item()

            # Adjust learning rate
            self.scheduler.step(training_loss)                   
            #self.lambda_kai_scheduler.step(epoch+1)

            train_log.append(training_loss)
            val_log.append(validation_loss)
            
            print('Epoch:', epoch+1, 'Training loss:', training_loss, 'Validation loss:',validation_loss)    # Average loss per epoch

        return train_log, val_log
                