# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 01:49:35 2025

@author: gaura
"""

from beta_vae import Beta_VAE
from dataset import Dsprites_Dataset
import torch
from torch.utils.data import DataLoader
from elbo_loss import Elbo_Loss
import pickle
from train_amp_utils import train, plot_losses
from torch.optim.lr_scheduler import CosineAnnealingLR




if __name__ == '__main__':
    model =  Beta_VAE(z_dim = 10)
    
    b_size = 128
    lr = 1e-4
    beta= 6
    num_epochs = 50
    
    directory = r"D:\work\projects\beta_VAE_from_scratch"  
    train_data = Dsprites_Dataset(directory, split = 'train')
    val_data = Dsprites_Dataset(directory, split = 'val')

    train_loader = DataLoader(train_data , batch_size = b_size ,shuffle = True)
    val_loader = DataLoader(val_data , batch_size = b_size , shuffle = False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loss_fn = Elbo_Loss(beta = beta)
    
    #checkpoint = torch.load(r'checkpoints\yolox_s_mosaic_vfocal_std_resample.pt', map_location=device)
    #model_state_dict = checkpoint['model_state']
    #opt_state_dict = checkpoint['optimizer_state']
    #min_val_loss = checkpoint['val_loss']
    #last_epoch = checkpoint['epoch']
    
    
    
    
    model = model.to(device)
    #model.load_state_dict(model_state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)#, last_epoch = last_epoch+15)
    
    
    
    trained_model, (train_loss_record, val_loss_record) = train(model, train_loader, val_loader, loss_fn , optimizer, device,
              epochs=num_epochs, validate_every_epoch=True, val_batches_limit=None,
              train_batches_limit=None, best_val_loss = None, checkpoint_path=r'checkpoints\b_VAE.pt', scheduler=None,
              early_stopping_patience=30, log_every=120, warmup = 10)
    
    plot_losses(train_loss_record ,val_loss_record)
    
    
    
   


