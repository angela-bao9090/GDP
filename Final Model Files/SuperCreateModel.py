from sklearn.preprocessing import StandardScaler
import time
import joblib

from SuperSKLModel import SuperSklearn as SuperSKLM






class SklearnModel(SuperSKLM):
    '''
    SuperClass for creating/loading and using supervised machine learning models from Scikit-Learn for making predictions and evaluating performance



    Attributes:
        model            - The loaded Scikit-Learn model                                              - sklearn model (created in subclasses)
        threshold        - The decision/cutoff boundary (likelihood of fraud that is flagged as such) - float (created in subclasses)
        scalar           - The scalar used to scale any data                                          - preprocessing object
        y_pred           - List to store predicted labels                                             - list
        y_true           - List to store true labels for evaluation                                   - list
        testing          - Flag to track if model is currently being tested on                        - bool
        features         - Features for input data                                                    - ndarray
        target           - Target labels for input data                                               - ndarray
        targetless       - Flag indicating if the input data does not have targets                    - bool
        X_file           - Features from the dataset file                                             - ndarray
        y_file           - Target labels from the dataset file                                        - ndarray
        cm               - A confusion matrix containing test results                                 - ndarray
        paddedcm         - Placeholder for cm to ensure there is no missing data                      - ndarray
        accuracy         - Calculated accuracy of the model on the testing data                       - float
        train_input      - The training data                                                          - list of lists
        test_input       - The testing data                                                           - list of lists
        model_type       - String for filename                                                        - str
        timestamp        - String for uniqueness of filename                                          - str
        filename         - String that the saved file will be named                                   - str
        to_save          - Data to be contained in saved file                                         - dict

    Methods:
        __init__ - Initializes the UseModelSklearn class by loading the model from a file
        
            Parameters:
                model_file -  Path to the model file - str
        
        resetStored - Resets the stored prediction and true labels lists
        
        loadTargetless - Loads input data for targetless predictions
        
            Parameters:
                data_points - Collection of entries relating to the dataset the model is trained on without targets - list of lists
        
        loadTargeted - Loads labeled input data for prediction
        
            Parameters:
                data_points - Collection of entries relating to the dataset the model is trained on with targets - list of lists
                existScalar - Indicates whether or not a scalar already exists                                   - bool
        
        predict - Makes predictions on the provided input data (either targetless or labeled)
        
        test - Tests the model on the loaded input data (if labeled), calculating accuracy, f1 score (weighted and unweighted), and confusion matrix
        
        train - Trains the model using the training dataset
        
        commenceTraining - Runs the training and testing phases, outputs the results as confusion matrices at the end
        
        saveModel - Saves the model, its threshold, and type into a file with a timestamped name
    '''
    
    def __init__(self, train_input, test_input, threshold = 0.5):
        super().__init__()
        self.scalar = StandardScaler()
        
        self.train_input = train_input
        self.test_input = test_input
        self.threshold= threshold
        
    def train(self):
        self.model.fit(self.features, self.target)
        
    def commenceTraining(self):
        self.loadTargeted(self.train_input, False)
        self.train()
        
        self.resetStored()
        
        self.loadTargeted(self.test_input)
        self.test()
        
        self.resetStored()
        
    def saveModel(self):         
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.file_name = f"{self.model_type}Model_{self.timestamp}.joblib"
        
        self.to_save = {
            "model": self.model,
            "threshold": self.threshold,
            "scalar": self.scalar
        }   
        
        joblib.dump(self.to_save, self.file_name)
    