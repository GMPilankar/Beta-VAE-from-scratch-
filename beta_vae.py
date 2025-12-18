# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 00:07:44 2025

@author: gaura
"""

import torch
import torch.nn as nn
from model_utils import Encoder , Decoder


class Beta_VAE(nn.Module):
    def __init__(self, z_dim = 10):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        
        
    def forward(self, x):
        mu  , log_var  = self.encoder(x)
        z = mu + (torch.exp(log_var * 0.5) * torch.randn_like(log_var))
        px = self.decoder(z)
        
        return  mu , log_var, px

        