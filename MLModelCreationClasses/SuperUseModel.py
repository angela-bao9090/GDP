import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np
import joblib
from sklearn.preprocessing import PowerTransformer
from sklearn.ensemble import IsolationForest
from DatasetReader import DatasetReader, TargetlessDatasetReader
from Plot import plotCM, plotP






class UseModelPyTorch:
    '''
    Class for loading and using PyTorch models for making predictions and evaluating performance
    
    

    Attributes:
        model            - The loaded PyTorch model                                                   - torch.nn.Module
        model_type       - Type of the model                                                          - str
        threshold        - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float
        y_pred           - List to store predicted labels                                             - list
        y_true           - List to store true labels for evaluation                                   - list
        testing          - Flag to track if model is currently being tested on                        - bool
        input_features   - Features for input data                                                    - tensor
        input_target     - Target labels for input data                                               - tensor
        input_targetless - Flag indicating if the input data does not have targets                    - bool
        file_dataset     - Dataset object for reading data from file                                  - DatasetReader
        file_loader      - DataLoader object for batching and loading data                            - DataLoader
        targetless       - Flag indicating whether the file has no targets                            - bool



    Methods:
        __init__ - Initializes the model by loading the model from a file and setting up necessary variables
        
            Parameters:
                model_file -  Path to the model file (.pth for PyTorch) - str
        
        loadModel - Loads the model from the provided file, sets the model to evaluation mode, and initializes thresholds
        
            Parameters:
                model_file -  Path to the model file (.pth for PyTorch) - str
        
        setThreshold - Sets the decision threshold for classification
        
            Parameters:
                threshold - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float
                
        resetStored - Resets the stored lists for predictions and true labels
        
        loadTargetlessDatabaseFile - Loads a targetless dataset from a CSV file
        
            Parameters:
                csv_file - The path to the CSV file containing the dataset without targets - str
                
        loadDatabaseFile - Loads a labeled dataset from a CSV file
        
            Parameters:
                csv_file - The path to the CSV file containing the dataset with targets - str
        
        predictOnFile - Makes predictions on the data loaded from a file and generates confusion matrix
                
        iteratePredict - Iterates over batches to make predictions

            Parameters:
                X - The current batch of features from the dataset file - Torch.tensor
                
        testOnFile - Tests the model on a file with labeled data, calculates accuracy and confusion matrix
        
        loadTargetlessInput - Loads a data points for targetless input prediction

            Parameters:
                Data_points - Collection of entries relating to the dataset the model is trained on without targets - list of lists
        
        loadInput - Loads data points with both features and target for prediction
        
            Parameters:
                Data_points - Collection of entries relating to the dataset the model is trained on with targets - list of lists
                
        predictOnInput - Makes predictions on the loaded input data (either with or without targets)
        
        testOnInput - Tests the model on the loaded input data (if labeled), calculating accuracy and confusion matrix
    '''
    
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
        self.model_type = self.model_data["model_type"]
        self.scalar = self.model_data["scalar"]
             
    def setThreshold(self, threshold):
        self.threshold = threshold
        
    def resetStored(self):
        self.y_pred = []
        self.y_true = []
    
    def loadTargetlessDatabaseFile(self, csv_file):  # Store appropriate file variables in self
        self.file_dataset = TargetlessDatasetReader(csv_file = csv_file, scalar = self.scalar)
        
        self.file_loader = DataLoader(self.file_dataset, batch_size = 64, shuffle = False)  # Don't set shuffle true!!!
        
        self.targetless = True
        
    def loadDatabaseFile(self, csv_file):
        self.file_dataset = DatasetReader(csv_file = csv_file, scalar = self.scalar)
        
        self.file_loader = DataLoader(self.file_dataset, batch_size = 64, shuffle = False)
        
        self.targetless = False
       
    def predictOnFile(self):
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
                plotP(self.cm)
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
            plotCM([self.cm], ["Model Test Results"])          
        
    def loadTargetlessInput(self, data_points):  # Ensure that input is a 2D array (each row is a separate data point)
        self.features = torch.tensor(data_points, dtype=torch.float32).to(self.device).unsqueeze(0)
        self.scalar.transform(self.input_features)
        
        self.input_targetless = True
        
    def loadInput(self, data_points):  # Ensure that input is a 2D array (each row is a separate data point)
        input_tensor = torch.tensor(data_points, dtype=torch.float32)

        self.input_features = input_tensor[:, :-1]  # All columns except the last one
        self.input_target = input_tensor[:, -1]    # Last column is the target
        
        self.scalar.transform(self.input_features)
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
                if self.cm.shape[1] <2:
                    self.padded_cm = np.zeros((1, 2), dtype=int)  
                    self.padded_cm[0, int(self.y_pred[0])] = self.cm[0, 0]     
                    self.cm = self.padded_cm
                plotP(self.cm)
                self.resetStored()

    def testOnInput(self):
        if self.input_targetless:
            pass
        else:
            self.testing = True
            
            self.predictOnInput()
            self.y_true.extend(self.input_target)
            self.cm = confusion_matrix(self.y_true, self.y_pred)
            
            if self.cm.shape[0] < 2 or self.cm.shape[1] < 2:
                # Initialize a 2x2 confusion matrix with zeros
                self.padded_cm = np.zeros((2, 2), dtype=int)
            
                if self.cm.shape == (1, 1):
                    self.padded_cm[int(self.y_true[0]), int(self.y_pred[0])] = self.cm[0, 0]  # Fill in the single value into the [0, 0] position of the padded matrix
                elif self.cm.shape == (1, 2):  # If it's a 1x2 matrix (this can happen if we predicted both classes, but one class was not predicted)
                    self.padded_cm[int(self.y_true[0]), 0] = self.cm[0, 0]
                    self.padded_cm[int(self.y_true[0]), 1] = self.cm[0, 1]
                elif self.cm.shape == (2, 1):  # If it's a 2x1 matrix (this can happen if both classes were predicted, but not both)
                    self.padded_cm[0, int(self.y_pred[0])] = self.cm[0, 0]
                    self.padded_cm[1, int(self.y_pred[0])] = self.cm[1, 0]
                
                self.cm = self.padded_cm
                
            self.testing = False
            
            self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()
            print(f"Accuracy: {self.accuracy * 100:.2f}%")
            plotCM([self.cm], ["Model Test Results"])
    
    

class UseModelSklearn:
    '''
    Class for loading and using supervised machine learning models from Scikit-Learn for making predictions and evaluating performance



    Attributes:
        model            - The loaded Scikit-Learn model                                              - sklearn model
        model_type       - The type of the model                                                      - str
        threshold        - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float
        y_pred           - List to store predicted labels                                             - list
        y_true           - List to store true labels for evaluation                                   - list
        testing          - Flag to track if model is currently being tested on                        - bool
        input_features   - Features for input data                                                    - ndarray
        input_target     - Target labels for input data                                               - ndarray
        input_targetless - Flag indicating if the input data does not have targets                    - bool
        file_dataset     - Dataset object for reading data from file                                  - DatasetReader
        targetless       - Flag indicating whether the file has no targets                            - bool
        X_file           - Features from the dataset file                                             - ndarray
        y_file           - Target labels from the dataset file                                        - ndarray

    Methods:
        __init__ - Initializes the UseModelSklearn class by loading the model from a file.
        
            Parameters:
                model_file -  Path to the model file (.pth for PyTorch) - str
        
        loadModel - Loads the trained model and relevant metadata (threshold, model type).
        
            Parameters:
                model_file -  Path to the model file (.pth for PyTorch) - str
        
        setThreshold - Allows setting a new threshold for classification.
        
            Parameters:
                threshold - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float
        
        resetStored - Resets the stored prediction and true labels lists.
        
        loadTargetlessDatabaseFile - Loads a dataset without target labels for prediction.
        
            Parameters:
                csv_file - The path to the CSV file containing the dataset without targets - str
        
        loadDatabaseFile - Loads a labeled dataset with both features and target labels.
        
            Parameters:
                csv_file - The path to the CSV file containing the dataset with targets - str
        
        predictOnFile - Makes predictions on a dataset loaded from a file.
        
        testOnFile - Tests the model on a labeled dataset, calculating accuracy and confusion matrix.
        
        loadTargetlessInput - Loads input data for targetless predictions.
        
            Parameters:
                Data_points - Collection of entries relating to the dataset the model is trained on without targets - list of lists
        
        loadInput - Loads labeled input data for prediction.
        
            Parameters:
                Data_points - Collection of entries relating to the dataset the model is trained on with targets - list of lists
        
        predictOnInput - Makes predictions on the provided input data (either targetless or labeled).
        
        testOnInput - Tests the model on the loaded input data (if labeled), calculating accuracy and confusion matrix.
    '''
    
    def __init__(self, model_file):
        self.loadModel(model_file)

        self.y_pred = []
        self.y_true = []
        self.testing = False
        
        self.input_features = None 
        self.input_target = None
        
        self.input_targetless = True
        
        self.file_dataset = None
        
        self.targetless = True
        
    def loadModel(self, model_file):
        self.model_data = joblib.load(model_file)
        self.model = self.model_data["model"]
        

        self.threshold =  self.model_data["threshold"]
        self.model_type = self.model_data["model_type"]
        self.scalar = self.model_data["scalar"]
        
    def setThreshold(self, threshold):
        self.threshold = threshold
    
    def resetStored(self):
        self.y_pred = []
        self.y_true = []
        
    def loadTargetlessDatabaseFile(self, csv_file):  # Store appropriate file variables in self
        self.features = TargetlessDatasetReader(csv_file = csv_file, scalar = self.scalar)
        
        self.targetless = True
    
    def loadDatabaseFile(self, csv_file):
        self.file_dataset = DatasetReader(csv_file = csv_file, scalar = self.scalar)
        
        self.X_file = self.file_dataset.features
        self.y_file = self.file_dataset.target
        
        self.targetless = False
        
    def predictOnFile(self):
        if self.file_dataset == None:
            pass
        else:
            if not self.testing:
                self.resetStored()
                
            self.y_prob = self.model.predict_proba(self.X_file)[:, 1]
            self.y_pred.extend((self.y_prob >= self.threshold).astype(int))
            self.y_true.extend(self.y_file)
    
        
            if not self.testing:
                self.cm = confusion_matrix([0]*len(self.y_pred), self.y_pred)
                plotP(self.cm)
                self.resetStored()   
    
    def testOnFile(self):
        if self.targetless:
            pass
        else:
            self.testing = True
            
            self.predictOnFile()
            self.cm = confusion_matrix(self.y_true, self.y_pred)
            
            self.testing = False

            self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()
            print(f"Accuracy: {self.accuracy * 100:.2f}%")
            plotCM([self.cm], ["Model Test Results"])         
            
    def loadTargetlessInput(self, data_points):  # Ensure that input is a 2D array (each row is a separate data point)
        self.input_features = np.array(data_points) 
        self.input_features = self.scalar.transform(self.input_features) 
        self.input_targetless = False
            
    def loadInput(self, data_points):  # Ensure that input is a 2D array (each row is a separate data point)
        self.data_points = np.array(data_points)
        self.input_features = self.data_points[:, :-1]
        self.input_features = self.scalar.transform(self.input_features)
        self.input_target = self.data_points[:, -1]
        self.input_targetless = False
        
    def predictOnInput(self):
        if self.input_features is None:
            pass 
        else:
            if not self.testing:
                self.resetStored()
                
            y_prob = self.model.predict_proba(self.input_features)[:, 1]  
            print(f"probability of fraud: {y_prob}")
            self.y_pred.extend((y_prob >= self.threshold).astype(int))
            
          
            if not self.testing:
                self.cm = confusion_matrix([0]*len(self.y_pred), self.y_pred)
                if self.cm.shape[1] <2:
                    self.padded_cm = np.zeros((1, 2), dtype=int)  
                    self.padded_cm[0, int(self.y_pred[0])] = self.cm[0, 0]     
                    self.cm = self.padded_cm
                plotP(self.cm)
                self.resetStored()




    def testOnInput(self):
        if self.input_targetless:
            pass
        else:
            self.testing = True
            
            self.predictOnInput()
            self.y_true.extend(self.input_target)
            self.cm = confusion_matrix(self.y_true, self.y_pred)
            if self.cm.shape[0] < 2 or self.cm.shape[1] < 2:
                # Initialize a 2x2 confusion matrix with zeros
                self.padded_cm = np.zeros((2, 2), dtype=int)
            
                if self.cm.shape == (1, 1):
                    self.padded_cm[int(self.y_true[0]), int(self.y_pred[0])] = self.cm[0, 0]  # Fill in the single value into the [0, 0] position of the padded matrix
                elif self.cm.shape == (1, 2):  # If it's a 1x2 matrix (this can happen if we predicted both classes, but one class was not predicted)
                    self.padded_cm[int(self.y_true[0]), 0] = self.cm[0, 0]
                    self.padded_cm[int(self.y_true[0]), 1] = self.cm[0, 1]
                elif self.cm.shape == (2, 1):  # If it's a 2x1 matrix (this can happen if both classes were predicted, but not both)
                    self.padded_cm[0, int(self.y_pred[0])] = self.cm[0, 0]
                    self.padded_cm[1, int(self.y_pred[0])] = self.cm[1, 0]
                
                self.cm = self.padded_cm
            self.testing = False
            
            self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()
            print(f"Accuracy: {self.accuracy * 100:.2f}%")
            plotCM([self.cm], ["Model Test Results"])

class UseModelIsolationForest:

    def merchantInput(self, df, merchant, day, model, scalar, iso_model):
        assumed_fraud = False
        
        anomaly_threshold = 0.7
        # min threshold for anomaly - if anomaly score is greater than this, we should possibly flag it as fraud
        max_fraud_threshold = 0.8
        mean_fraud_threshold = 0.6
        odd_hour_ratio_threshold = 0.4


        # model is the model to calculate individual fraud probabilities, and then iso_model is the isolation forest

        daily_transactions = df[(df['merchant'] == merchant) & (df['day'] == day)]

        features = daily_transactions.drop(columns=['merchant', 'day', 'label'], errors='ignore')
        features = self.scalar.transform(features)  # Scale features
        probs = self.model.predict_proba(features)[:, 1]  # Calculates raw fraud probabilities
        max_fraud = np.max(probs)
        mean_fraud = np.mean(probs)
        median_fraud = np.median(probs)
        std_fraud = np.std(probs)
        num_transactions = len(probs)

        if max_fraud >= max_fraud_threshold or mean_fraud >= mean_fraud_threshold:
            assumed_fraud = True

        amounts = group['amount'].values
        times = group['time'].values

        # need to see what the actual column names are for amount spent and timem
        odd_hours = (times >= 23) | (times <= 6)    
        odd_hour_count = np.sum(odd_hours)
        odd_hour_ratio = odd_hour_count/num_transactions

        if odd_hour_ratio >= odd_hour_ratio_threshold:
            assumed_fraud = True


        mean_spend = np.mean(amounts)
        max_spend = np.max(amounts)
        std_dvt_spend = np.std(amounts)

        features = np.array([[num_transactions, mean_fraud, max_fraud, std_fraud, median_fraud, mean_spend, max_spend, std_dvt_spend, odd_hour_count]])

        anomaly_score = 1 - iso_model.score_samples(features)[0]
        # approximate probability for being fraud 

        if anomaly_score >= anomaly_threshold:
            assumed_fraud = True
        