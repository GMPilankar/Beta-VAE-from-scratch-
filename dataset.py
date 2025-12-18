# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 21:30:26 2025

@author: gaura
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

def train_val_test_split(arrays, train=0.8, val=0.1, test=0.1, seed=None):
    """
    arrays: list or tuple of numpy arrays with same first dimension
    returns: (train_arrays, val_arrays, test_arrays)
    """
    assert abs(train + val + test - 1.0) < 1e-6
    n = arrays[0].shape[0]

    for arr in arrays:
        assert arr.shape[0] == n, "All arrays must have same number of rows"

    if seed is not None:
        np.random.seed(seed)

    perm = np.random.permutation(n)

    n_train = int(train * n)
    n_val   = int(val * n)

    idx_train = perm[:n_train]
    idx_val   = perm[n_train:n_train + n_val]
    idx_test  = perm[n_train + n_val:]

    train_arrays = [arr[idx_train] for arr in arrays]
    val_arrays   = [arr[idx_val] for arr in arrays]
    test_arrays  = [arr[idx_test] for arr in arrays]

    return train_arrays, val_arrays, test_arrays

class Dsprites_Dataset(Dataset):
    def __init__(self, root_dir, split = 'train'):
        data_file = os.path.join(root_dir , "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        with np.load(data_file) as data:
            self.imgs = data['imgs']
            self.latent_values = data['latents_values']
            self.latent_classes = data['latents_classes']
        
        arrays = [self.imgs, self.latent_values, self.latent_classes]
        
        train_list ,val_list, test_list = train_val_test_split(arrays, train=0.8, val=0.1, test=0.1, seed=0)
        
        if split == 'train':
            self.imgs ,self.latent_values ,self.latent_classes = train_list
        elif split == 'val':
            self.imgs ,self.latent_values ,self.latent_classes = val_list
        elif split == 'test':
            self.imgs ,self.latent_values ,self.latent_classes = test_list
        else:
            raise ValueError(f"Invalid split '{split}'. Must be one of {{'train', 'val', 'test'}}")
            
    
    def __len__(self,):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img = np.expand_dims(img, axis=0)
        latent_val = np.expand_dims(self.latent_values[idx], axis = 0)
        
        return torch.as_tensor(img, dtype = torch.float32) , torch.as_tensor(latent_val, dtype = torch.float32)
        


if __name__ == '__main__':
    root_dir = r'D:\work\projects\beta_VAE_from_scratch'
    train_dataset = Dsprites_Dataset(root_dir, 'train')
    val_dataset = Dsprites_Dataset(root_dir, 'val')
    test_dataset = Dsprites_Dataset(root_dir, 'test')
    
    

