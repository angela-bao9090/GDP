# Todo: Comment code 

# Importing libraries
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pandas as pd



class DatasetReader(Dataset): # Inherits from imported Dataset
    '''
    Class for accessing a database, standardizing the data(excluding the target column), accessing the data for machine learning 
    
    Parameters:
        file path - location of csv file
        transform - (if needed) a transform on the data
    '''
    def __init__(self, csv_file, transform=None): 
        # Access file, split columns into features and target, apply standard scalar and transform
        
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
        self.features = self.data.iloc[:, :-1].values  # All columns except for the last one
        self.target = self.data.iloc[:, -1].values  # Just the last column
        
        self.features = StandardScaler().fit_transform(self.features) # Standardising the features and the applying transform

    def __len__(self): 
        # Returns the size of the dataset
        
        return len(self.data)
    
    def size(self): 
        # Returns the number of columns in the features
        
        return self.features.shape[1]

    def __getitem__(self, idx): 
        # Returns the features and target in a useful form
        
        sample_features = torch.tensor(self.features[idx], dtype=torch.float32)
        sample_target = torch.tensor(self.target[idx], dtype=torch.float32)

        return sample_features, sample_target