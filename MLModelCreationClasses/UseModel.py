# Todo: Sklearn loading class | combined loading class | comments
# !!! need to implement some type of queue/stack to work with multiple incoming data points and remove them once classified !!!

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from DatasetReader import DatasetReader, TargetlessDatasetReader
from Plot import ConfusionMatrix as PlotCM, Predictions as PlotP


class UseModelPyTorch: # Only works for loading PyTorch models - other models will be loadable soon 
    def __init__(self, model_file):
        self.loadModel(model_file)

        self.y_pred = []
        self.y_true = []
        self.testing = False
        
        self.input_features = None 
        self.input_target = None
        
        self.input_targetless = True
        
        self.file_dataset = None
        self.file_loader = None
        
        self.targetless = True
        
    def loadModel(self, model_file):
        self.model_data = torch.load(model_file, weights_only=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model_data["model"]
        self.model = self.model.to(self.device)
        
        self.model.eval()

        self.threshold =  self.model_data["threshold"]
        
    def setThreshold(self, threshold):
        self.threshold = threshold
        
    def resetStored(self):
        self.y_pred = []
        self.y_true = []
    
    def loadTargetlessDatabaseFile(self, csv_file): # Store appropriate file variables in self
        self.file_dataset = TargetlessDatasetReader(csv_file = csv_file)
        
        self.file_loader = DataLoader(self.file_dataset, batch_size = 64, shuffle = False) # Don't set shuffle true!!!
        
        self.targetless = True
        
    def loadDatabaseFile(self, csv_file):
        self.file_dataset = DatasetReader(csv_file = csv_file)
        
        self.file_loader = DataLoader(self.file_dataset, batch_size = 64, shuffle = False)
        
        self.targetless = False
       
    def predictOnFile(self):
        # Make test() function (for both Pytorch and sklearn) an importable class. - remember some data needs to be accessible for plotting
        # Use here and in MLModel
        # Store appropriate self. variables related to made predictions
        # possibly make super super models
        if self.file_loader == None:
            pass
        
        else:
            if not self.testing:
                self.resetStored()
                
            with torch.no_grad():  # No gradients needed during inference
                if self.targetless:
                    for X in self.file_loader:
                        self.iteratePredict(X)
                else:
                    for X,_ in self.file_loader:
                        self.iteratePredict(X)
                    
            if not self.testing:
                self.cm = confusion_matrix([0]*len(self.y_pred), self.y_pred)
                PlotP(self.cm)
                self.resetStored()   
        
    def iteratePredict(self,X):
        X = X.to(self.device)

        self.outputs = self.model(X)
        self.predicted = (torch.sigmoid(self.outputs) >= self.threshold).float()  # Apply sigmoid (map to output value)

        self.y_pred.extend(self.predicted.cpu().numpy())      
        
    def testOnFile(self):
        if self.targetless:
            pass
        else:
            self.testing = True
            
            self.predictOnFile()
            self.y_true.extend(self.file_dataset.target)
            self.cm = confusion_matrix(self.y_true, self.y_pred)
            
            self.testing = False

            self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()
            print(f"Accuracy: {self.accuracy * 100:.2f}%")
            PlotCM([self.cm], ["Model Test Results"])          
        
    def loadTargetlessDataPoint(self, data_point):
        self.features = torch.tensor(data_point, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        self.input_targetless = True
        
    def loadDataPoint(self, data_point): # Input 2D array - each internal array is a single datapoint
        input_tensor = torch.tensor(data_point, dtype=torch.float32)
        
        self.input_features = input_tensor[:, :-1]  # All columns except the last one
        self.input_target = input_tensor[:, -1]    # Last column is the target
        
        self.input_targetless = False
        
    def predictOnInput(self):
        if self.input_features == None:
            pass 
        else:
            if not self.testing:
                self.resetStored()
        
            with torch.no_grad():  # No gradients needed during inference
                self.model.eval()  # Ensure the model is in evaluation mode
                outputs = self.model(self.input_features)
                predicted = (torch.sigmoid(outputs) >= self.threshold).float()
                self.y_pred.extend(predicted.cpu().numpy())

            if not self.testing:
                self.cm = confusion_matrix([0]*len(self.y_pred), self.y_pred)
                PlotP(self.cm)
                self.resetStored()

    def testOnInput(self):
        if self.input_targetless:
            pass
        else:
            self.testing = True
            
            self.predictOnInput()
            self.y_true.extend(self.input_target)
            self.cm = confusion_matrix(self.y_true, self.y_pred)
            
            self.testing = False
            
            self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()
            print(f"Accuracy: {self.accuracy * 100:.2f}%")
            PlotCM([self.cm], ["Model Test Results"])
    

        
        
        

def main(): # Test code
    model = UseModelPyTorch("NNModel_20250325_222016.pth")
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/creditcard.csv"
    print(f"threshold: {model.threshold}")
    model.loadDatabaseFile(test_file)
    model.testOnFile()
    model.testOnFile()
    model.predictOnFile()
    model.testOnFile()
    
    model.loadDataPoint([[0]*31, [1]*31])
    model.predictOnInput()





        