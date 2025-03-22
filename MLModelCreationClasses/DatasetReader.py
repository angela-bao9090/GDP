# Todo: Comment code 

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DatasetReader(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        self.features = self.data.iloc[:, :-1].values  
        self.target = self.data.iloc[:, -1].values 
        
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_features = torch.tensor(self.features[idx], dtype=torch.float32)
        
        sample_label = torch.tensor(self.target[idx], dtype=torch.float32)
        
        if self.transform:
            sample_features = self.transform(sample_features)

        return sample_features, sample_label