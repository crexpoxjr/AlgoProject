"""
A data loader that generates LOB-shaped data from the fi2010 dataset.

Shapes match the real pipeline:
    X: (N, seq_len, 40) float32     y: (N,) int64 in {0: Up, 1: Stationary, 2: Down}
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

class fi2010LOBDataset(Dataset):
    def __init__(self, seq_len=100, n_features=40, dataset_name="Train_Dst_NoAuction_ZScore_CF_1.txt"):
        ## yoink the data and slap it into self.x and self.y so then everything else can grab data
        raw = np.loadtxt(dataset_name)
        lob_features = raw[:n_features].T
        lob_targets = raw[144] -1

        X = []
        Y = []

        for i in range(len(lob_features) - seq_len):
            X.append(lob_features[i:i+seq_len])
            Y.append(lob_targets[i+seq_len])

        self.x = torch.from_numpy(np.array(X).astype(np.float32))
        self.y = torch.from_numpy(np.array(Y).astype(np.int64))

        print(self.x.shape)
        print(self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def get_fi2010_dataloaders(batch_size=64, seq_len=100, n_train=2000, n_val=400, n_test=400, seed=0, dataset_name="Train_Dst_NoAuction_ZScore_CF_1.txt"):
    # seed kept for compatibility with synthetic loader
    dataset = fi2010LOBDataset(seq_len, dataset_name="Train_Dst_NoAuction_ZScore_CF_1.txt")

    n_total = n_train + n_val + n_test
    train_end = int(n_train/n_total * len(dataset))
    val_end = train_end + int(n_val/n_total * len(dataset))

    train = Subset(dataset, range(0, train_end))
    val = Subset(dataset, range(train_end, val_end))
    test = Subset(dataset, range(val_end, len(dataset)))
    
    return (DataLoader(train, batch_size=batch_size, shuffle=False), 
            DataLoader(val, batch_size=batch_size),
            DataLoader(test, batch_size=batch_size))

def get_dataloaders(config, dataset_name="Train_Dst_NoAuction_ZScore_CF_1.txt"):
    return get_fi2010_dataloaders(config["batch_size"], config["seq_len"], seed=config["seed"], dataset_name="Train_Dst_NoAuction_ZScore_CF_1.txt")