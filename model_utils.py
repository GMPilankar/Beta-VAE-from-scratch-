# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 23:10:58 2025

@author: gaura
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, z_dim = 10):
        super().__init__()
        self.z_dim = z_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU()
            )
        
        self.linear_mu = nn.Linear(4096 , z_dim)
        self.linear_log_var = nn.Linear(4096 , z_dim)
    
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = torch.flatten(conv_out , 1)
        mu = self.linear_mu(conv_out)
        log_var = self.linear_log_var(conv_out)
        
        return mu , log_var
    

class Decoder(nn.Module):
    def __init__(self,z_dim = 10 ):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4096)
        self.decoder = nn.Sequential(
            # Layer 1: 4x4 -> 8x8
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            
            # Layer 4: 32x32 -> 64x64 (Final Layer)
            nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Force pixels to [0, 1] range
            )
    
    def forward(self, z):
        z = self.linear(z)
        z = z.view(-1, 256, 4, 4)
        x = self.decoder(z)
        
        return x 




    

