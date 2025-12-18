# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 13:54:20 2025

@author: gaura
"""


import torch.nn as nn
import torch
#import math
import torch.nn.functional as F


class Elbo_Loss(nn.Module):
    def __init__(self, beta = 1):
        super().__init__()
        self.beta = beta
    
    
    def forward(self,mu , log_var, recon_x , x):
        # elbo = recon_term - kl_term
        # maximize elbo (same as minimizing -elbo loss)
        neg_recon_term = F.binary_cross_entropy(recon_x, x, reduction='sum')
        
        kl_term = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        return neg_recon_term + self.beta * kl_term


if __name__ == '__main__':
    pass
    # inp = torch.ones(16, 1, 64,64)
    # target = torch.ones(16, 1, 64,64)
    
    # loss = F.binary_cross_entropy(inp ,target , reduction = 'none')
    # print(loss.shape)      