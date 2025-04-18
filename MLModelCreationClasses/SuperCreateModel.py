import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import time
import joblib
import numpy as np
from collections import defaultdict
import pandas as pd
from DatasetReader import DatasetReader
from Plot import plotCM
from sklearn.ensemble import IsolationForest





class TorchModel(nn.Module):
    '''
    Superclass for ML models from PyTorch - i.e. Regression, NeuralNetworks, etc.
    
    Attributes:
        train_file    - Location of the dataset file used for training                                  - str
        test_file     - Location of the dataset file used for testing                                   - str
        batch_size    - Number of samples per iteration before updating the model's weights             - int
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        train_loader  - DataLoader object for iterating through the training dataset                    - DataLoader
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        test_loader   - DataLoader object for iterating through the testing dataset                     - DataLoader
        model         - The machine learning model (e.g., Neural Network, Regression)                   - nn.Module
        loss_fn       - The loss function used to train the model                                       - torch.nn.Module
        optimizer     - The optimizer used to update the model’s weights                                - torch.optim.Optimizer
        device        - The device (CPU or GPU) on which the model runs                                 - torch.device
        epochs        - Number of iterations to train the model                                         - int
        running_loss  - The cumulative loss during one epoch of training                                - float
        train_loss    - The average training loss for one epoch                                         - float
        y_pred        - List of predictions made by the model during testing                            - list
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        file_name     - The filename for saving the model                                               - str
        to_save       - Dictionary containing the model and related information to be saved             - dict
        pred          - Predictions made by the model during training/testing                           - torch.Tensor
        loss          - The loss calculated during training or testing                                  - torch.Tensor
        outputs       - Raw output of the model before applying the threshold                           - torch.Tensor
        predicted     - Predictions after applying threshold or activation function (e.g., sigmoid)     - torch.Tensor
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        
        
        
    Methods:
        __init__ - Initializes the class with the training and test files, batch size, and threshold
            
            Parameters:
                train_file - location of dataset file to be used for training                           - str
                test_file  - location of dataset file to be used for testing                            - str
                batch_size - Number of samples per iteration before updating the models weights         - int (optional)
                threshold  - the decision/cutoff boundary (likelihood of fraud that is flagged as such) - float (optional)

        initModel - Placeholder method to initialize the model, loss function, optimizer, and device (Must be implemented in subclasses)
            
        train - Performs one epoch of training by iterating over the training dataset in batches. Updates model weights and calculates the loss
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases for multiple epochs and stores confusion matrices for each epoch
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, batch_size = 64, threshold = 0.5):
        super().__init__()
        self.train_file = train_file
        self.batch_size = batch_size
        
        self.train_dataset = DatasetReader(csv_file = self.train_file, undersample=True)
        self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
        self.scalar = self.train_dataset.getScalar()
        
        self.test_dataset = DatasetReader(csv_file = test_file, scalar = self.scalar)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = False)
        
        self.threshold = threshold

        
        # Example code for the subclass override
        #
        #    def __init__(self, train_file, test_file, hyperparameters)
        #        super().__init__(train_file, test_file)
        #        self.titles = [titles for graphs]
        #        self.initModel(hyperParameters) 
        
    def initModel(self): 
        # Load the model, choose loss function, choose optimizer, choose device to run on
        # Implemented in subclasses
        
        return NotImplemented
        
        #code in subclasses will look something like:
        #
        #   self.model = # Model used
        #   self.loss_fn = # Chosen Loss Function
        #   self.optimizer = # Optimizer
        #   self.epochs = # Number of iterations
        #
        #   self.device = # Devices to use for model (Server, CPU, etc. )
        #   self.model = self.model.to(self.device)  
        
    def train(self):
        self.model.train() # Set model to training mode
        
        self.running_loss = 0.0
        for _, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            self.pred = self.model(X)

            self.loss = self.loss_fn(self.pred.squeeze(), y)  
            
            self.optimizer.zero_grad()  
            self.loss.backward()
            self.optimizer.step()  

            self.running_loss += self.loss.item()
                
        self.train_loss = self.running_loss / len(self.train_loader)

    def test(self):
        self.model.eval()  # Set model to evaluation mode
        self.y_pred = []

        with torch.no_grad():  # No gradients needed during inference
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.outputs = self.model(X)
                self.predicted = (torch.sigmoid(self.outputs) >= self.threshold).float()  # Apply sigmoid (map to output value)

                self.y_pred.extend(self.predicted.cpu().numpy())

        self.cm = confusion_matrix(self.test_dataset.target, self.y_pred)

        self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()     
 
    def commenceTraining(self):
        self.cms = []
        for t in range(self.epochs):
            print(f"-------------------------------\nEpoch {t+1}\n-------------------------------")
            self.train()
            print(f"Train Loss: {self.train_loss:>7f}")
            self.train_dataset = DatasetReader(csv_file = self.train_file, undersample=True, scalar = self.scalar)
            self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True)
            self.test()
            print(f"Accuracy: {self.accuracy * 100:.2f}%")   
            self.cms.append(self.cm)
        
        plotCM(self.cms, self.titles)
        
    def saveModel(self): 
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_name = f"{self.model_type}Model_{self.timestamp}.pth"
        
        self.to_save = {
            "model": self.model,
            "threshold": self.threshold,
            "model_type": "PyTorch",
            "scalar": self.scalar
        }
        
        torch.save(self.to_save, self.file_name)
        
        
        
class SklearnModel:
    '''
    Superclass for ML models from SKLearn - i.e. Gradient Boosting, Random Forest, etc. 
    Note: threshold doesn't make any difference for Isolation Forest since it is an unsupervised model
    
    
    
    Attributes:
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        threshold     - The decision/cutoff boundary for classification                                 - float
        model         - The machine learning model (e.g., Random Forest, Gradient Boosting)             - sklearn model
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - str


        
    Methods:
        __init__ - Initializes the class with the training and test files, threshold, and datasets
            
            Parameters:
                train_file - Location of dataset file to be used for training                           - str
                test_file  - Location of dataset file to be used for testing                            - str
                threshold  - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float (optional)

        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.5):
        self.train_dataset = DatasetReader(csv_file = train_file, undersample=True)
        
        self.X_train = self.train_dataset.features
        self.y_train = self.train_dataset.target
        self.scalar = self.train_dataset.getScalar()
        
        self.test_dataset = DatasetReader(csv_file=test_file, scalar=self.scalar)
        self.X_test = self.test_dataset.features
        self.y_test = self.test_dataset.target
        
        self.threshold = threshold
        
        
        # Example code for subclass override
        #
        #    super().__init__(train_file, test_file, threshold)
        #    self.model = Model(hyperparameters) --- Will be added in subclasses overwrite of model
        #    self.titles = [graph title]ß
        
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        
    def test(self):
        if (self.supervised): # Is a supervised ML model
            self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
            self.y_pred = (self.y_prob >= self.threshold).astype(int)
            
        else: # Is an Unsupervised ML model - Since they only classify and dont give probabilities - Also why threshold doesn't do anything
            self.y_pred = self.model.predict(self.X_test)
            self.y_pred = [1 if label == 1 else 0 for label in self.y_pred] 

        self.cm = confusion_matrix(self.y_test, self.y_pred)
        
        self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()     
        
    def commenceTraining(self):        
        self.train()
        self.test()
        print(f"Accuracy: {self.accuracy * 100:.2f}%")  
        self.cms = [self.cm]
        plotCM(self.cms, self.titles)
        
    def tuning(self): # Not yet implemented 
        # Test a number of different hyperparameters to determine the best selections for a model
        
        return NotImplemented
    
        # Possible code for tuning a random forest - would need specified for each ML model based on hyperparameters
        #
        #    self.classifier = RandomForestClassifier(    
        #        random_state=random_state,
        #        class_weight='balanced',  
        #        oob_score=True,           
        #        n_jobs=-1                
        #    )
        #    
        #    self.param_grid = {
        #        'n_estimators': [50, 100, 200],
        #        'max_depth': [10, 20, 30, None],
        #        'min_samples_split': [5, 10],
        #        'min_samples_leaf': [1, 4],
        #    }
        #    
        #    self.grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.param_grid, cv=5, verbose=2, n_jobs=-1)
        #    self.grid_search.fit(self.X_train, self.y_train)
        #    print("Best hyperparameters:", self.grid_search.best_params_)
        #    
        #    self.model = self.grid_search.best_estimator_
        
    def saveModel(self):         
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_name = f"{self.model_type}Model_{self.timestamp}.joblib"
        
        self.to_save = {
            "model": self.model,
            "threshold": self.threshold,
            "model_type": "Sklearn",
            "scalar": self.scalar
        }   
        
        joblib.dump(self.to_save, self.file_name)

class IsolationForestModel:
    # write comments


    def buildDailySummary(self, df, model, scalar):
        daily_transactions = defaultdict(lambda: defaultdict(dict))
        grouped = df.groupby(['merchant', 'day'])
        for (merchant, day), group in grouped:
            features = group.drop(columns=['merchant', 'day', 'label'], errors='ignore')
            features = scalar.transform(features) 
            # not sure if scaling is needed, but have seen it in some other functions
            # now calculating relevant statistics
            probs = model.predict_proba(features)[:, 1]
            count = len(probs)
            max_fraud = np.max(probs)
            mean_fraud = np.mean(probs)
            std_dvt_fraud = np.std(probs)
            median_fraud = np.median(probs)

            amounts = group['amount'].values
            times = group['time'].values

            # need to see what the actual column names are for amount spent and timem

            odd_hours = (times >= 23) | (times <= 6)    
            odd_hour_count = np.sum(odd_hours)

            mean_spend = np.mean(amounts)
            max_spend = np.max(amounts)
            std_dvt_spend = np.std(amounts)

            daily_transactions[merchant][day] = {
                'num_transactions': count,
                'mean_fraud': mean_fraud,
                'max_fraud': max_fraud,
                'std_dvt_fraud': std_dvt_fraud,
                'median_fraud': median_fraud,
                'mean_spend': mean_spend,
                'max_spend': max_spend,
                'std_dvt_spend': std_dvt_spend,
                'odd_hour_transactions': odd_hour_count

                # We can add more things here but need to decide on what is relevant
            }

        return daily_transactions

    def daily_to_df(self, daily_transactions):
        rows = []

        for merchant, days in daily_transactions.items():
            for day, stats in days.items():
                # Create a flat dictionary for each merchant/day
                row = {'merchant': merchant, 'day': day}
                row.update(stats)  # Add all the summary stats
                rows.append(row)

        summary_df = pd.DataFrame(rows)
        return summary_df

    def trainIsolationForest(self, summary_df):
        contamination = 0.01
        features = summary_df.drop(columns = (['merchant', 'day']))
        iso_model = IsolationForest(contaminatino = contamination, n_jobs = -1)
        iso_model.fit(features)
        return iso_model, features