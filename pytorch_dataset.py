# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 13:24:24 2020

@author: groes
"""
from torch.utils.data import Dataset
import torch

class MakePytorchData(Dataset):
    def __init__(self, X, y): # data should be given as tensors
        self.X = torch.from_numpy(X.values).float() 
        self.y = torch.from_numpy(y.values).float() 
        self.num_samples = len(X)
        # Data loading
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.num_samples