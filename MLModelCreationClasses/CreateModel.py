# File containing classes for following types of ML algorithms:
# Neural Network (using Logistic Regression)
# Logistic Regression
# XGBoost (eXtreme Gradient Boosting)
# Gradient Boosting
# Random Forest
# Isolation Forest



import torch
from torch import nn
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest

from SuperModels import TorchModel as TM, SklearnModel as SKLM

    
    
class NeuralNetworkModel(TM):
    '''
    Class for creating a Neural Network which uses Logistic Regression - Inherits from TorchModel in the SuperModels file
    
    Parameters:
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        batch_size - number of samples per iteration before updating the models weights
        threshold - the decision/cutoff boundary (likelihood of fraud that is flagged as such)
        learning_rate - how much model weights are adjusted wrt loss gradient - determines step size at each iteration
        epochs - number of times training dataset is passed through
        momentum - allows for faster convergence, but may overshoot
        weight_decay - adds penalty to loss function to stop learnig overly complex/large weights (makes models more simple and less chance of overfitting)
        hiden_layer_sizes - sizes of each layer of the Neural Network (also determines number of layers)
        dropout_rate - scaling of weights when dropout (seting some neurons to zero during each training step) is applied 
        activation_fn - mathematical operation applied to output of a neuron - introduces non-linearity to the network 
    '''
    
    def __init__(self, train_file, test_file, batch_size=64, threshold = 0.75,
                 learning_rate=1e-3, epochs=8, momentum=0.8, weight_decay=0.0,
                 hidden_layer_sizes=[8, 8], dropout_rate=0.2, activation_fn=nn.ReLU):
        # Use superclass __init__(), create titles for plots, initiate model
        
        super().__init__(train_file, test_file, batch_size, threshold)
        self.model_type = "NN" # For filename when saving
        self.titles = []
        for i in range(epochs):
            self.titles.append('Neural Network - Epoch')
        self.initModel(learning_rate, epochs, momentum, weight_decay, hidden_layer_sizes, dropout_rate, activation_fn)
        
    def initModel(self, learning_rate, epochs, momentum, weight_decay, hidden_layer_sizes, dropout_rate, activation_fn):
        # Make the model with the hyperparameters, choose optimizer
        
        self.layers = []
        
        # Hidden layers
        prev_size = self.train_dataset.size()
        for hidden_size in hidden_layer_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.layers.append(activation_fn())  # Apply the activation function
            if dropout_rate > 0:
                self.layers.append(nn.Dropout(dropout_rate))  # Add dropout if specified
            prev_size = hidden_size
        self.layers.append(nn.Linear(prev_size, 1))  
        
        self.model = nn.Sequential(*self.layers)
        self.loss_fn = nn.BCEWithLogitsLoss()  

        #self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        self.epochs = epochs
        
        
class LogisticRegressionModel(TM):
    '''
    Class for creating a Logistic Regression ML algorithm - Inherits from TorchModel in the SuperModels file
    
    Parameters:
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        batch_size - number of samples per iteration before updating the models weights
        threshold - the decision/cutoff boundary (likelihood of fraud that is flagged as such)
        learning_rate - how much model weights are adjusted wrt loss gradient - determines step size at each iteration
        epochs - number of times training dataset is passed through
        momentum - allows for faster convergence, but may overshoot
        weight_decay - adds penalty to loss function to stop learnig overly complex/large weights (makes models more simple and less chance of overfitting)   
    '''
    
    def __init__(self, train_file, test_file, batch_size=64, threshold = 0.7, 
                 learning_rate=1e-3, epochs=5, momentum=0.9, weight_decay=0.0):   
        # Use superclass __init__(), create titles for plots, initiate model
         
        super().__init__(train_file, test_file, batch_size, threshold)
        self.model_type = "LR" # For filename when saving
        self.titles = []
        for i in range(epochs):
            self.titles.append('Logistic Regression - Epoch')
        self.initModel(learning_rate, epochs, momentum, weight_decay)
        
    def initModel(self, learning_rate, epochs, momentum, weight_decay):
        # Make the model with the hyperparameters, choose optimizer
        
        self.model = nn.Linear(self.train_dataset.size(), 1) 
        self.loss_fn = nn.BCEWithLogitsLoss()  
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
            
        self.epochs = epochs
        
        
class XGBoostModel(SKLM):
    '''
    Class for creating a eXtreme Gradient Boosting ML algorithm - Inherits from Sklearn in the SuperModels file
    
    Parameters:
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        threshold - the decision/cutoff boundary (likelihood of fraud that is flagged as such)
        n_estimators - number of boosting rounds to be used (higher value usually means more complex model)
        learning_rate - how much model weights are adjusted wrt loss gradient - determines step size at each iteration
        max depth - max depth of individual trees
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.6, n_estimators = 100, learning_rate = 0.1, max_depth = 3):
        # Use superclass __init__(), initiate model, specify supervised ML algorithm, create titles for plot
        
        super().__init__(train_file, test_file, threshold)
    
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,  
            learning_rate=learning_rate,  
            max_depth=max_depth,   
        )
        self.supervised = True
        self.model__type = "XGB"
        self.titles = ["XGBoost"]
        
        
        
class GradientBoostingMachineModel(SKLM):
    '''
    Class for creating a Gradient Boosting ML algorithm - Inherits from Sklearn in the SuperModels file
    
    Parameters:
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        threshold - the decision/cutoff boundary (likelihood of fraud that is flagged as such)
        n_estimators - number of boosting rounds to be used (higher value usually means more complex model)
        learning_rate - how much model weights are adjusted wrt loss gradient - determines step size at each iteration
        max depth - max depth of individual trees
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.6, n_estimators=100, learning_rate=0.1, max_depth=3):
        # Use superclass __init__(), initiate model, specify supervised ML algorithm, create titles for plot
        
        super().__init__(train_file, test_file, threshold)
        
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        self.supervised = True
        self.model_type = "GB"
        self.titles = ["Gradient Boosting Machine"]
        
        
        
class RandomForestModel(SKLM):
    '''
    Class for creating a Random Forest ML algorithm - Inherits from Sklearn in the SuperModels file
    
    Parameters:
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        threshold - the decision/cutoff boundary (likelihood of fraud that is flagged as such)
        n_estimators - number of boosting rounds to be used (higher value usually means more complex model)
        max depth - max depth of individual trees
        min_samples_split - min number of samples to split an internal node
        min_samples_leaf - minimum samples required to be at a leaf node
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.6, n_estimators = 50, max_depth = 20, min_samples_split = 5, min_samples_leaf = 1):
        # Use superclass __init__(), initiate model, specify supervised ML algorithm, create titles for plot
        
        super().__init__(train_file, test_file, threshold)

        self.model = RandomForestClassifier(
            oob_score=True,         
            n_jobs=-1,                 
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
        self.supervised = True
        self.model_type = "RF"
        self.titles = ["Random Forest"]



class IsolationForestModel(SKLM):
    '''
    Class for creating an Isolation Forest ML algorithm - Inherits from Sklearn in the SuperModels file
    
    Parameters:
        train_file - location of dataset file to be used for training
        test_file  - location of dataset file to be used for testing
        contamination - expected proportion of outliers in training data
        n_estimators - number of boosting rounds to be used (higher value usually means more complex model)
    '''
    
    def __init__(self, train_file, test_file, contamination = "auto", n_estimators = 50): 
        # Use superclass __init__(), initiate model, specify NOT a supervised ML algorithm, create titles for plot
        
        super().__init__(train_file, test_file, 1) # 1 is a placeholder value since Isolation Forest doesn't use a threshold
        
        self.model = IsolationForest(
            n_estimators=n_estimators, 
            contamination=contamination,
            n_jobs=-1 
        )
        self.supervised = False
        self.model_type = "IF"
        self.titles = ["Isolation Forest"]

        
        
        
        
def main(): # Test code
    train_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/balanced.csv"
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/creditcard.csv"
    
    print("_________________________________________________________________________")
    print(" Neural Network using logisstic regression")
    model = NeuralNetworkModel(train_file, test_file)  
    model.commenceTraining() 
    model.saveModel()
    
    print("_________________________________________________________________________")
    print("Logistic Regression")
    model = LogisticRegressionModel(train_file, test_file)
    model.commenceTraining()
    
    print("_________________________________________________________________________")
    print("XGBoost")
    model = XGBoostModel(train_file, test_file, 0.6, 50, 0.005, 3, 73)
    model.commenceTraining()
    
    print("_________________________________________________________________________")
    print("Gradient Boosting Machine")
    model = GradientBoostingMachineModel(train_file, test_file, 0.99)
    model.commenceTraining()
    model.saveModel()
    
    print("_________________________________________________________________________")
    print("Random Forest")
    model = RandomForestModel(train_file, test_file) 
    model.commenceTraining()
    
    print("_________________________________________________________________________")
    print("Isolation Forest")
    train_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/creditcard.csv"
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/balanced.csv"
    model = IsolationForestModel(train_file, test_file)
    model.commenceTraining()
    
