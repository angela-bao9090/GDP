import torch
from torch import nn
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

from SuperCreateModel import TorchModel as TM, SklearnModel as SKLM

    



    
class TorchNeuralNetworkModel(TM):
    '''
    Class for creating a Neural Network which uses Logistic Regression - Inherits from TorchModel in the SuperCreateModel file
    
    
    
    Attributes:
        train_file    - Location of the dataset file used for training                                  - str
        test_file     - Location of the dataset file to be used for testing                             - str
        batch_size    - Number of samples per iteration before updating the model's weights             - int
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        train_loader  - DataLoader object for iterating through the training dataset                    - DataLoader
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        test_loader   - DataLoader object for iterating through the testing dataset                     - DataLoader
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - nn.Module
        loss_fn       - The loss function used to train the model                                       - torch.nn.Module
        optimizer     - The optimizer used to update the model’s weights                                - torch.optim.Optimizer
        device        - The device on which the model runs (CPU or GPU)                                 - torch.device
        epochs        - Number of epochs for training the model                                         - int
        running_loss  - The cumulative loss during one epoch of training                                - float
        train_loss    - The average training loss for one epoch                                         - float
        y_pred        - List of predictions made by the model during testing                            - list
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        file_name     - The filename for saving the model                                               - str
        to_save       - Dictionary containing the model and related information to be saved             - dict
        pred          - Predictions made by the model during training/testing                           - torch.Tensor
        loss          - The loss calculated during training                                             - torch.Tensor
        outputs       - Raw output of the model before applying the threshold                           - torch.Tensor
        predicted     - Predictions after checking against threshold                                    - torch.Tensor
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        
        
    
    Methods:
        __init__ - Initializes the neural network with given parameters and prepares the model for training
        
            Parameters:
                train_file        - location of dataset file to be used for training                                                                                     - str
                test_file         - location of dataset file to be used for testing                                                                                      - str
                batch_size        - number of samples per iteration before updating the models weights                                                                   - int (optional)
                threshold         - the decision/cutoff boundary (likelihood of fraud that is flagged as such)                                                           - float (optional)
                learning_rate     - how much model weights are adjusted wrt loss gradient (determines step size at each iteration)                                       - float (optional)
                epochs            - number of times training dataset is passed through                                                                                   - int (optional)
                momentum          - allows for faster convergence, but may overshoot                                                                                     - float (optional)
                weight_decay      - adds penalty to loss function to stop learnig overly complex/large weights (makes models more simple and less chance of overfitting) - float (optional)
                hiden_layer_sizes - sizes of each layer of the Neural Network (also determines number of layers)                                                         - list of int (optional)
                dropout_rate      - scaling of weights when dropout (seting some neurons to zero during each training step) is applied                                   - float (optional)
                activation_fn     - mathematical operation applied to output of a neuron (introduces non-linearity to the network)                                       - callable (optional)           

        initModel - Initializes the neural network model with specified hyperparameters and optimizer settings
        
            Parameters:
                learning_rate     - how much model weights are adjusted wrt loss gradient (determines step size at each iteration)                                       - float 
                epochs            - number of times training dataset is passed through                                                                                   - int 
                momentum          - allows for faster convergence, but may overshoot                                                                                     - float 
                weight_decay      - adds penalty to loss function to stop learnig overly complex/large weights (makes models more simple and less chance of overfitting) - float 
                hiden_layer_sizes - sizes of each layer of the Neural Network (also determines number of layers)                                                         - list of int 
                dropout_rate      - scaling of weights when dropout (seting some neurons to zero during each training step) is applied                                   - float 
                activation_fn     - mathematical operation applied to output of a neuron (introduces non-linearity to the network)                                       - callable           

        train - Placeholder for the training function that inherits from the superclass and fine-tunes the model
        
        test - Placeholder for the testing function that evaluates the trained model using the test dataset
        
        commenceTraining - Runs the training and testing phases for multiple epochs and stores confusion matrices for each epoch
        
        predict - Placeholder method that should be implemented to allow the model to make predictions on new data
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, batch_size=64, threshold = 0.7,
                 learning_rate=4e-4, epochs=14, momentum=0.95, weight_decay=0.0,
                 hidden_layer_sizes=[3,2], dropout_rate=0.5, activation_fn=nn.ReLU):
        
        super().__init__(train_file, test_file, batch_size, threshold)
        self.model_type = "NN" # For filename when saving
        self.titles = []
        for i in range(epochs):
            self.titles.append('Neural Network - Epoch')
        self.initModel(learning_rate, epochs, momentum, weight_decay, hidden_layer_sizes, dropout_rate, activation_fn)
        
    def initModel(self, learning_rate, epochs, momentum, weight_decay, hidden_layer_sizes, dropout_rate, activation_fn):
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
        

        
class TorchLogisticRegressionModel(TM):
    '''
    Class for creating a Logistic Regression ML algorithm - Inherits from TorchModel in the SuperCreateModel file
    
    
    
    Attributes:
        train_file    - Location of the dataset file used for training                                  - str
        test_file     - Location of the dataset file used for testing                                   - str
        batch_size    - Number of samples per iteration before updating the model's weights             - int
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        train_loader  - DataLoader object for iterating through the training dataset                    - DataLoader
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        test_loader   - DataLoader object for iterating through the testing dataset                     - DataLoader
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - nn.Module
        loss_fn       - The loss function used for training (Binary Cross-Entropy with Logits)          - nn.Module
        optimizer     - The optimizer used to update model weights (Stochastic Gradient Descent)        - torch.optim.Optimizer
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
        loss          - The loss calculated during training                                             - torch.Tensor
        outputs       - Raw output of the model before applying the threshold                           - torch.Tensor
        predicted     - Predictions after checking against threshold                                    - torch.Tensor
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
    
 
        
    Methods:
        __init__ - Initializes the logistic regression model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                                                                      - str
                test_file     - Location of dataset file to be used for testing                                                                                       - str
                batch_size    - Number of samples per iteration before updating the model's weights                                                                   - int (optional)
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)                                                            - float (optional)
                learning_rate - How much model weights are adjusted wrt loss gradient (determines step size at each iteration)                                        - float (optional)
                epochs        - Number of times training dataset is passed through                                                                                    - int (optional)
                momentum      - Allows for faster convergence, but may overshoot                                                                                      - float (optional)
                weight_decay  - Adds penalty to loss function to stop learning overly complex/large weights (makes models more simple and less chance of overfitting) - float (optional)

        initModel - Initializes the logistic regression model with specified hyperparameters and optimizer settings
        
            Parameters:
                learning_rate - How much model weights are adjusted wrt loss gradient (determines step size at each iteration)                                        - float 
                epochs        - Number of times training dataset is passed through                                                                                    - int 
                momentum      - Allows for faster convergence, but may overshoot                                                                                      - float
                weight_decay  - Adds penalty to loss function to stop learning overly complex/large weights (makes models more simple and less chance of overfitting) - float 
                
        train - Inherits the training method from the superclass, allowing for the training of the logistic regression model
        
        test - Inherits the testing method from the superclass, evaluating the trained model using the test dataset
        
        commenceTraining - Runs the training and testing phases for multiple epochs and stores confusion matrices for each epoch
        
        predict - Placeholder method that should be implemented to allow the model to make predictions on new data
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, batch_size=64, threshold = 0.7, 
                 learning_rate=1e-4, epochs=14, momentum=0.95, weight_decay=0.15):   
         
        super().__init__(train_file, test_file, batch_size, threshold)
        self.model_type = "LR" # For filename when saving
        self.titles = []
        for i in range(epochs):
            self.titles.append('Logistic Regression - Epoch')
        self.initModel(learning_rate, epochs, momentum, weight_decay)
        
    def initModel(self, learning_rate, epochs, momentum, weight_decay):
        
        self.model = nn.Linear(self.train_dataset.size(), 1) 
        self.loss_fn = nn.BCEWithLogitsLoss()  
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
            
        self.epochs = epochs
        
        
        
class NeuralNetwork(SKLM):
    '''
    Class for creating a Neural Network (Multilayer Perceptron) ML algorithm - Inherits from SklearnModel in the SuperModels file



    Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - MLPClassifier
        supervised    - Indicates whether the model is supervised (True for this model)                 - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - str



    Methods:
        __init__ - Initializes the Neural Network model with given parameters and prepares the model for training
        
            Parameters:
                train_file         - location of dataset file to be used for training                                               - str
                test_file          - location of dataset file to be used for testing                                                - str
                threshold          - the decision/cutoff boundary (likelihood of fraud that is flagged as such)                     - float (optional)
                hidden_layer_sizes - the number of neurons in each hidden layer (tuple of integers)                                 - tuple (optional)
                activation_fn      - mathematical operation applied to output of a neuron - introduces non-linearity to the network - str (optional)
                solver             - the optimizer algorithm ('adam', 'sgd', etc.)                                                  - str (optional)
                alpha              - regularization term to prevent overfitting (L2 penalty)                                        - float (optional)
                batch_size         - number of samples per iteration before updating the models weights                             - str or int (optional)
                learning_rate      - how much model weights are adjusted wrt loss gradient - determines step size at each iteration - str (optional)
                learning_rate_init - initial learning rate for the solver                                                           - float (optional)
                max_iter           - maximum number of iterations for the solver to converge                                        - int (optional)
                early_stopping     - whether to stop training when validation score is not improving                                - bool (optional)
                momentum           - allows for faster convergence, but may overshoot                                               - float (optional)
                n_iter_no_change   - number of iterations with no improvement before stopping                                       - int (optional)
        
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning(Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
        '''
        
    def __init__(self, train_file, test_file, threshold=0.97, 
                 hidden_layer_sizes=(2,2,2), activation_fn='relu', solver='adam',
                 alpha=0.0001, batch_size='auto', learning_rate='adaptive', 
                 learning_rate_init=0.003, max_iter=500, early_stopping=False,
                 momentum=0.9, n_iter_no_change=10):
        
        super().__init__(train_file, test_file, threshold)
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation_fn,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            momentum=momentum,
            n_iter_no_change=n_iter_no_change
        )
        
        self.supervised = True
        self.model_type = "NN"
        self.titles = ["Neural Network"]

            

class LogisticRegressionModel(SKLM):
    '''
    Class for creating a Logistic Regression ML algorithm - Inherits from Sklearn in the SuperModels file
    
    
    
    Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - LogisticRegression
        supervised    - Indicates whether the model is supervised (True for this model)                 - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - str



    Methods:
        __init__ - Initializes the Logistic Regression model with given parameters and prepares the model for training
                
            Parameters:
                    train_file    - Location of dataset file to be used for training                           - str
                    test_file     - Location of dataset file to be used for testing                            - str
                    threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float (optional)
                    C             - Regularization strength; smaller values specify stronger regularization    - float (optional)
                    max_iter      - Maximum number of iterations for the solver to converge                    - int (optional)
                    solver        - Optimization algorithm                                                     - str (optional)
                    penalty       - Regularization type                                                        - str (optional)
                    tol           - Tolerance for stopping criteria (smaller values = stricter stopping)       - float (optional)
                    class_weight  - Weights associated with classes to handle imbalanced classes               - str or dict (optional)
                    fit_intercept - Whether to include an intercept in the model                               - bool (optional)
                
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning(Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.91, C=5e-3, max_iter=200, solver='saga', penalty='l1', tol=1e-8, class_weight='balanced', fit_intercept=False):
        super().__init__(train_file, test_file, threshold)
    
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter, 
            solver=solver,
            penalty=penalty, 
            tol=tol,
            class_weight=class_weight, 
            fit_intercept=fit_intercept 
        )
        
        self.supervised = True
        self.model_type = "LR"
        self.titles = ["Logistic Regression"]
        


class NaiveBayes(SKLM):
    '''
    Class for creating a Naive Bayes ML algorithm - Inherits from Sklearn in the SuperModels file



        Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - GaussianNB
        supervised    - Indicates whether the model is supervised (True for this model)                 - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - str



    Methods:
        __init__ - Initializes the Naive Bayes model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                  - str
                test_file     - Location of dataset file to be used for testing                                   - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)        - float (optional)
                var_smoothing - Variance smoothing parameter to prevent division by zero in Gaussian distribution - float (optional)
                priors        - The class prior probabilities (default assumes imbalanced classes)                - list of floats (optional)
        
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning (Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold=0.98, var_smoothing=1e-9, priors=[0.999,0.001]):
        super().__init__(train_file, test_file, threshold)
        self.model = GaussianNB(var_smoothing=var_smoothing, priors=priors)
        self.supervised = True
        self.model_type = "NaiveBayes"
        self.titles = ["Naive Bayes"]
        


class GradientBoostingMachineModel(SKLM):
    '''
    Class for creating a Gradient Boosting ML algorithm - Inherits from Sklearn in the SuperModels file



    Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - GradientBoostingClassifier
        supervised    - Indicates whether the model is supervised (True for this model)                 - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - str



    Methods:
        __init__ - Initializes the Gradient Boosting model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                                                   - str
                test_file     - Location of dataset file to be used for testing                                                                    - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)                                         - float (optional)
                n_estimators  - Number of boosting rounds to be used (higher value usually means more complex model)                               - int (optional)
                learning_rate - How much model weights are adjusted with respect to the loss gradient (determines the step size at each iteration) - float (optional)
                max_depth     - Maximum depth of individual trees                                                                                  - int (optional)
        
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning (Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.95, n_estimators=100, learning_rate=0.1, max_depth=5):
        super().__init__(train_file, test_file, threshold)
        
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
        self.supervised = True
        self.model_type = "GB"
        self.titles = ["Gradient Boosting Machine"]
        
          
        
class XGBoostModel(SKLM):
    '''
    Class for creating an eXtreme Gradient Boosting ML algorithm - Inherits from Sklearn in the SuperModels file



    Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - XGBClassifier
        supervised    - Indicates whether the model is supervised (True for this model)                 - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - int



    Methods:
        __init__ - Initializes the eXtreme Gradient Boosting model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                                               - str
                test_file     - Location of dataset file to be used for testing                                                                - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)                                     - float (optional)
                n_estimators  - Number of boosting rounds to be used (higher value usually means more complex model)                           - int (optional)
                learning_rate - How much model weights are adjusted with respect to loss gradient - determines the step size at each iteration - float (optional)
                max_depth     - Maximum depth of individual trees                                                                              - int (optional)
        
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning (Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.95, n_estimators = 100, learning_rate = 0.1, max_depth = 5):
        super().__init__(train_file, test_file, threshold)
    
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,  
            learning_rate=learning_rate,  
            max_depth=max_depth,   
        )
        self.supervised = True
        self.model_type = "XGB"
        self.titles = ["XGBoost"]
        


class CatBoostModel(SKLM):
    '''
    Class for creating an CatBoost ML algorithm - Inherits from Sklearn in the SuperModels file



    Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - XGBClassifier
        supervised    - Indicates whether the model is supervised (True for this model)                 - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - int



    Methods:
        __init__ - Initializes the eXtreme Gradient Boosting model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                                               - str
                test_file     - Location of dataset file to be used for testing                                                                - str
                threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)                                     - float (optional)
                iterations    - The number of boosting iterations for the model training                                                       - int (optional)
                learning_rate - How much model weights are adjusted with respect to loss gradient - determines the step size at each iteration - float (optional)
                depth         - The maximum depth of the decision trees used in boosting                                                       - int (optional)   
                l2_leaf_reg   - L2 regularization term for the leaf nodes to prevent overfitting                                               - float (optional)
                subsample     - Fraction of the data to be used for training each tree (controls overfitting)                                  - float (optional)
        
        
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning (Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold=0.95, iterations=500, learning_rate=0.009, depth=6, 
                 l2_leaf_reg=3, subsample=1):
        
        super().__init__(train_file, test_file, threshold)
        
        self.model = CatBoostClassifier(
            iterations=iterations, 
            learning_rate=learning_rate, 
            depth=depth,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            verbose=0  
        )
        
        self.supervised = True
        self.model_type = 'CatBoost'
        self.titles = ["CatBoost"]
        
                 
        
class RandomForestModel(SKLM):
    '''
    Class for creating a Random Forest ML algorithm - Inherits from Sklearn in the SuperModels file



    Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - RandomForestClassifier
        supervised    - Indicates whether the model is supervised (True for this model)                 - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
        threshold     - The decision/cutoff boundary (likelihood of fraud that is flagged as such)      - float
        train_dataset - The training dataset object after reading the training file                     - DatasetReader
        X_train       - Features for training data                                                      - ndarray
        y_train       - Target labels for training data                                                 - ndarray
        test_dataset  - The testing dataset object after reading the test file                          - DatasetReader
        X_test        - Features for test data                                                          - ndarray
        y_test        - Target labels for test data                                                     - ndarray
        y_prob        - Predicted probabilities for positive class (if applicable)                      - ndarray
        y_pred        - Predicted labels (0 or 1) after applying threshold (for classification)         - ndarray
        cm            - Confusion matrix generated during the testing phase                             - ndarray
        accuracy      - The accuracy of the model on the test dataset                                   - float
        cms           - List of confusion matrices collected during training and testing for each epoch - list
        timestamp     - Timestamp used to generate unique filenames for saved models                    - str
        file_name     - Filename for saving the model with timestamp                                    - str



    Methods:
        __init__ - Initializes the Random Forest model with given parameters and prepares the model for training
        
            Parameters:
                train_file        - Location of dataset file to be used for training                                     - str
                test_file         - Location of dataset file to be used for testing                                      - str
                threshold         - The decision/cutoff boundary (likelihood of fraud that is flagged as such)           - float (optional)
                n_estimators      - Number of boosting rounds to be used (higher value usually means more complex model) - int (optional)
                max_depth         - Maximum depth of individual trees                                                    - int (optional)
                min_samples_split - Minimum number of samples required to split an internal node                         - int (optional)
                min_samples_leaf  - Minimum number of samples required to be at a leaf node                              - int (optional)
        
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning (Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, threshold = 0.6, n_estimators = 50, max_depth = 20, min_samples_split = 5, min_samples_leaf = 1): 
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



    Attributes:
        model_type    - Type of model (used for saving filename)                                        - str
        titles        - List of titles for the plots (one for each epoch)                               - list
        model         - The machine learning model                                                      - IsolationForest
        supervised    - Indicates whether the model is supervised (False for this model)                - bool
        train_file    - Location of dataset file to be used for training                                - str
        test_file     - Location of dataset file to be used for testing                                 - str
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
        __init__ - Initializes the Isolation Forest model with given parameters and prepares the model for training
        
            Parameters:
                train_file    - Location of dataset file to be used for training                                     - str
                test_file     - Location of dataset file to be used for testing                                      - str
                contamination - Expected proportion of outliers in the training data                                 - float (or 'auto', optional, default = 'auto')
                n_estimators  - Number of boosting rounds to be used (higher value usually means more complex model) - int (optional, default = 50)
        
        train - Trains the model using the training dataset
        
        test - Evaluates the model's performance on the test dataset and calculates accuracy and confusion matrix
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        tuning (Not Implemented) - Placeholder method to be implemented for hyperparameter tuning to find the best model configuration
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_file, test_file, contamination = "auto", n_estimators = 50): 
        super().__init__(train_file, test_file, 1)  # 1 is a placeholder value since Isolation Forest doesn't use a threshold
        
        self.model = IsolationForest(
            n_estimators=n_estimators, 
            contamination=contamination,
            n_jobs=-1 
        )
        self.supervised = False
        self.model_type = "IF"
        self.titles = ["Isolation Forest"]





                
def main():  # Test code
    train_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTrain.csv"
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTest.csv"
    
    '''print("_________________________________________________________________________")
    print(" Neural Network using logisstic regression - PyTorch")
    model = TorchNeuralNetworkModel(train_file, test_file)  
    model.commenceTraining() 
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Logistic Regression - PyTorch")
    model = TorchLogisticRegressionModel(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()'''
    
    print("_________________________________________________________________________")
    print("Neural Network")
    model = NeuralNetwork(train_file, test_file)
    model.commenceTraining() 
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Logistic Regression")
    model = LogisticRegressionModel(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Naives Bayes")
    model = NaiveBayes(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Gradient Boosting Machine")
    model = GradientBoostingMachineModel(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("XGBoost")
    model = XGBoostModel(train_file, test_file)
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("CatBoost")
    model = CatBoostModel(train_file, test_file)
    model.commenceTraining() 
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Random Forest")
    model = RandomForestModel(train_file, test_file) 
    model.commenceTraining()
    #model.saveModel()
    
    print("_________________________________________________________________________")
    print("Isolation Forest")
    train_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTrain.csv"
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTest.csv"
    model = IsolationForestModel(train_file, test_file)
    model.commenceTraining()
    
    
    
    
    
    
if __name__ == "__main__":
    main()