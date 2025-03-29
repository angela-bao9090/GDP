import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import time
import joblib

from DatasetReader import DatasetReader
from Plot import ConfusionMatrix as PlotCM



class TorchModel(nn.Module):
    '''
    Superclass for ML models from PyTorch - i.e. Regression, NeuralNetworks, etc.

    Parameters :
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        batch_size - Number of samples per iteration before updating the models weights
        threshold - the decision/cutoff boundary (likelihood of fraud that is flagged as such)
    '''
    
    def __init__(self, train_file, test_file, batch_size = 64, threshold = 0.5):
        # initialise the datasets for use, store the threshold and batch size
        # Subclass' function will also initialise the model by calling initModel and will generate titles for plotted graphs
        
        super().__init__()
        
        self.train_dataset = DatasetReader(csv_file = train_file)
        self.train_loader = DataLoader(self.train_dataset, batch_size = batch_size, shuffle = True)
        
        self.test_dataset = DatasetReader(csv_file = test_file)
        self.test_loader = DataLoader(self.test_dataset, batch_size = batch_size, shuffle = False)
        
        self.threshold = threshold

        
        # Example code for the subclass override
        '''
        def __init__(self, train_file, test_file, hyperparameters)
            super().__init__(train_file, test_file)
            self.titles = [titles for graphs]
            self.initModel(hyperParameters) 
        '''
        
    def initModel(self): 
        # Load the model, choose loss function, choose optimizer, choose device to run on
        # Implemented in subclasses
        
        return NotImplemented
        '''
        code in subclasses will look something like:
    
        self.model = # Model used
        self.loss_fn = # Chosen Loss Function
        self.optimizer = # Optimizer
        self.epochs = # Number of iterations
        
        self.device = # Devices to use for model (Server, CPU, etc. )
        self.model = self.model.to(self.device)  
        '''

    def train(self):
        # One epoch of training  - runs an iteration for each batch (in which loss is updated)
        
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
        # Test model using test data, generate confusion matrix to be plotted later, calculate accuracy
        
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
        # Begin training model - run train and test for each epoch, output the results as confusion matrices at the end
        
        self.cms = []
        for t in range(self.epochs):
            print(f"-------------------------------\nEpoch {t+1}\n-------------------------------")
            self.train()
            print(f"Train Loss: {self.train_loss:>7f}")
            self.test()
            print(f"Accuracy: {self.accuracy * 100:.2f}%")   
            self.cms.append(self.cm)
        
        PlotCM(self.cms, self.titles)
        
    def predict(self): # Not yet implemented
        # Allow the model to make predictions on new data
        
        return NotImplemented
        
    def saveModel(self): 
        # Allow the model to be saved
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_name = f"{self.model_type}Model_{self.timestamp}.pth"
        
        self.to_save = {
            "model": self.model,
            "threshold": self.threshold,
            "model_type": "PyTorch"
        }
        
        torch.save(self.to_save, self.file_name)
        
        
        
class SklearnModel:
    '''
    Superclass for ML models from SKLearn - i.e. Gradient Boosting, Random Forest, etc.

    Parameters :
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        threshold - the decision/cutoff boundary (likelihood of fraud that is flagged as such)
        
    Note: threshold doesn't make any difference for Isolation Forest since it is an unsupervised model
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.5):
        # initialise the datasets for use, store the threshold and batch size
        # Subclass' function will also initialise the model
        
        self.train_dataset = DatasetReader(csv_file=train_file)
        self.X_train = self.train_dataset.features
        self.y_train = self.train_dataset.target
        
        self.test_dataset = DatasetReader(csv_file=test_file)
        self.X_test = self.test_dataset.features
        self.y_test = self.test_dataset.target
        
        self.threshold = threshold
        
        
        # Example code for subclass override
        '''
        super().__init__(train_file, test_file, threshold)
        self.model = Model(hyperparameters) --- Will be added in subclasses overwrite of model
        self.titles = [graph title]
        '''
        
    def train(self):
        # Train model on training data
        
        self.model.fit(self.X_train, self.y_train)
        
    def test(self):
        # Test model using test data, generate confusion matrix to be plotted later, calculate accuracy 
        
        if (self.supervised): # Is a supervised ML model
            self.y_prob = self.model.predict_proba(self.X_test)[:, 1]
            self.y_pred = (self.y_prob >= self.threshold).astype(int)
            
        else: # Is an Unsupervised ML model - Since they only classify and dont give probabilities - Also why threshold doesn't do anything
            self.y_pred = self.model.predict(self.X_test)
            self.y_pred = [1 if label == 1 else 0 for label in self.y_pred] 

        self.cm = confusion_matrix(self.y_test, self.y_pred)
        
        self.accuracy = (self.cm[0, 0] + self.cm[1, 1]) / self.cm.sum()       
        
    def commenceTraining(self):
        # Begin training model - run train and test, output the results as confusion matrices at the end
        
        self.train()
        self.test()
        print(f"Accuracy: {self.accuracy * 100:.2f}%")  
        self.cms = [self.cm]
        PlotCM(self.cms, self.titles)
           
    def predict(self, X_new): # Not yet implemented
        # Allow model to make predictions on new data
        
        return NotImplemented
    
        # Possible code - not tested
        '''
        if self.transform:
            X_new = self.scaler.transform(X_new) 
        
        return self.model.predict(X_new)'''
        
    def tuning(self): # Not yet implemented 
        # Test a number of different hyperparameters to determine the best selections for a model
        
        return NotImplemented
    
        # Possible code for tuning a random forest - would need specified for each ML model based on hyperparameters
        '''
        self.classifier = RandomForestClassifier(    
            random_state=random_state,
            class_weight='balanced',  
            oob_score=True,           
            n_jobs=-1                
        )
        
        self.param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [1, 4],
        }
        
        self.grid_search = GridSearchCV(estimator=self.classifier, param_grid=self.param_grid, cv=5, verbose=2, n_jobs=-1)
        self.grid_search.fit(self.X_train, self.y_train)
        print("Best hyperparameters:", self.grid_search.best_params_)
        
        self.model = self.grid_search.best_estimator_'''
        
    def saveModel(self): 
        # Allow the model to be saved
        
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_name = f"{self.model_type}Model_{self.timestamp}.joblib"
        
        self.to_save = {
            "model": self.model,
            "threshold": self.threshold,
            "model_type": "Sklearn"
        }   
        
        joblib.dump(self.to_save, self.file_name)
