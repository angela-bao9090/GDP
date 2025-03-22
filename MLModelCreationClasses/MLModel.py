# Todo: fix inconsistent variable naming | comment code | remove repeated code | make Autoencoder | add (a possible) expected output | add more customisability to some of the models | try to universalise test results output

#file containing classes for types of ML algorithms
# Neural Network (using Logistic Regression)
# Logistic Regression
# XGBoost (eXtreme Gradient Boosting)
# Gradient Boosting
# Random Forest
# Isolated Forest

# apologies for the messy and hard to read code, this will be cleaned up in the next few days

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, log_loss

from DatasetReader import DatasetReader

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_size, train_file, test_file):
        super().__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.linear_relu_stack = self.linear_relu_stack.to(self.device)  
        
        self.train_dataset = DatasetReader(csv_file=train_file)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        
        self.test_dataset = DatasetReader(csv_file=test_file)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        self.loss_fn = nn.BCEWithLogitsLoss()  
        self.optimizer = torch.optim.SGD(self.linear_relu_stack.parameters(), lr=1e-3)
        
        self.epochs = 5

    def forward(self, x):
        logits = self.linear_relu_stack(x)  
        return logits  

    def train(self):
        size = len(self.train_loader.dataset)
        self.linear_relu_stack.train()
        
        running_loss = 0.0
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.linear_relu_stack(X)

            loss = self.loss_fn(pred.squeeze(), y)  

            self.optimizer.zero_grad()  
            loss.backward()  
            self.optimizer.step()  

            running_loss += loss.item()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
                
        return running_loss / len(self.train_loader)

    def test(self):
        count1 = 0
        count2 = 0
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        self.linear_relu_stack.eval()  
        test_loss, correct = 0, 0
        incorrect_predictions = []  
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.linear_relu_stack(X)  
                test_loss += self.loss_fn(logits.squeeze(), y).item()  

                predicted = (torch.sigmoid(logits) > 0.5).float() 

                incorrect_indices = (predicted.squeeze() != y).nonzero(as_tuple=True)[0]  
                
                for idx in incorrect_indices:
                    if y[idx].item() == 1: count1 += 1
                    else: count2 += 1
                    incorrect_predictions.append({
                        'index': idx.item(),
                        'true_label': y[idx].item(),
                        'predicted_label': predicted[idx].item(),
                        'logits': logits[idx].item()
                    })

                correct += (predicted.squeeze() == y).sum().item()

        test_loss /= num_batches  
        accuracy = 100 * correct / size  
        print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
        
        if incorrect_predictions:
            print("false negatives", count1, "/492")
            print("false positives:", count2)
        return test_loss, accuracy
        
    def commenceTraining(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = self.train()
            print(f"Train Loss: {train_loss:>7f}")
            test_loss, accuracy = self.test()

        print("Done!")
        
        
        
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, train_file, test_file):
        super(LogisticRegressionModel, self).__init__()
        
        self.model = nn.Linear(input_size, 1)  
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)  
        
        self.train_dataset = DatasetReader(csv_file=train_file)
        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        
        self.test_dataset = DatasetReader(csv_file=test_file)
        self.test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=False)
        
        self.loss_fn = nn.BCEWithLogitsLoss()  
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-3)
        
        self.epochs = 5

    def forward(self, x):
        logits = self.model(x)  
        return logits  

    def train_one_epoch(self):
        size = len(self.train_loader.dataset)
        self.model.train()
        
        running_loss = 0.0
        for batch, (X, y) in enumerate(self.train_loader):
            X, y = X.to(self.device), y.to(self.device)

            pred = self.model(X)

            loss = self.loss_fn(pred.squeeze(), y)  
            
            self.optimizer.zero_grad()  
            loss.backward()  
            self.optimizer.step() 

            running_loss += loss.item()

            if batch % 100 == 0:
                current = (batch + 1) * len(X)
                print(f"Loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
                
        return running_loss / len(self.train_loader)

    def test(self):
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        self.model.eval()  
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                logits = self.model(X)  
                test_loss += self.loss_fn(logits.squeeze(), y).item()  

                predicted = (torch.sigmoid(logits) > 0.5).float()  
                correct += (predicted.squeeze() == y).sum().item() 

        test_loss /= num_batches  
        accuracy = 100 * correct / size  
        print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss, accuracy
        
    def commenceTraining(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = self.train_one_epoch()
            print(f"Train Loss: {train_loss:>7f}")
            test_loss, accuracy = self.test()

        print("Done!")
        
        
        
# Below class includes a possible model to perform on single date input after testing and training
class XGBoostModel:
    def __init__(self, train_file, test_file, n_estimators = 100, learning_rate = 0.1, max_depth = 3, random_state = 42):
        self.train_dataset = DatasetReader(csv_file=train_file)
        self.test_dataset = DatasetReader(csv_file=test_file)
        
        self.X_train = self.train_dataset.features
        self.y_train = self.train_dataset.target
        self.X_test = self.test_dataset.features
        self.y_test = self.test_dataset.target
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,  
            learning_rate=learning_rate,  
            max_depth=max_depth,  
            random_state=random_state  
        )

        
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        
    def test(self):
        y_pred = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        return accuracy
    
    def commenceTraining(self):
        self.train()
        self.test()
    
    # Possible prediction algorithm - for new input data
    '''def predict(self, X_new):
        if self.transform:
            X_new = self.scaler.transform(X_new) 
        
        return self.model.predict(X_new)'''
        
        

class GradientBoostingMachineModel:
    def __init__(self, train_file, test_file, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        self.train_dataset = DatasetReader(csv_file=train_file)

        self.test_dataset = DatasetReader(csv_file=test_file)
        
        self.X_train = self.train_dataset.features
        self.y_train = self.train_dataset.target
        
        self.X_test = self.test_dataset.features
        self.y_test = self.test_dataset.target
        
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
        
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        
    def test(self):
        self.val_predictions = self.model.predict(self.X_test)
        self.val_accuracy = accuracy_score(self.y_test, self.val_predictions)
        print(f"Validation Accuracy: {self.val_accuracy * 100:.2f}%")

        self.test_predictions = self.model.predict(self.X_test)
        self.test_accuracy = accuracy_score(self.y_test, self.test_predictions)
        print(f"Test Accuracy: {self.test_accuracy * 100:.2f}%")

        print("Classification Report on Test Data:")
        print(classification_report(self.y_test, self.test_predictions))

        self.probabilities = self.model.predict_proba(self.X_test)[:, 1] 

        self.threshold = 0.5  
        self.predicted_anomalies = (self.probabilities > self.threshold).astype(int)

        print("Anomaly Detection Classification Report:")
        print(classification_report(self.y_test, self.predicted_anomalies))
        
    def commenceTraining(self):
        self.train()
        self.test()
        
        
# Below class includes a possible method for tuning hyperparameters
class RandomForestModel:
    def __init__(self, train_file, test_file, n_estimators = 50, max_depth = 20, min_samples_split = 5, min_samples_leaf = 1, random_state = 42):
        
        self.train_dataset = DatasetReader(csv_file=train_file)
        self.test_dataset = DatasetReader(csv_file=test_file)
        
        self.X_train = self.train_dataset.features
        self.y_train = self.train_dataset.target
        
        self.X_test = self.test_dataset.features
        self.y_test = self.test_dataset.target
        
        self.classifier = RandomForestClassifier(    
            random_state=random_state,
            class_weight='balanced',  
            oob_score=True,           
            n_jobs=-1                
        )
        
        self.model = RandomForestClassifier(
            random_state=42,
            oob_score=True,         
            n_jobs=-1,                 
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf
        )
    
    # Testing a mwthod of hyperparameter tuning - looks promising
    '''def tuning(self):
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
    
    def train(self):
        self.model.fit(self.X_train, self.y_train)
        
    def test(self):
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_prob = self.model.predict_proba(self.X_test)[:, 1]  
    
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.logloss = log_loss(self.y_test, self.y_pred_prob)

        print(f"Random Forest Model Accuracy: {self.accuracy * 100:.2f}%")
        print(f"Model Log Loss: {self.logloss:.4f}")

        print(f"OOB Score (training data): {self.model.oob_score_:.4f}")
        
    def commenceTraining(self):
        self.train()
        self.test()
        


class IsolationForestModel:
    def __init__(self, train_file, test_file, contamination = "auto", n_estimators = 100, random_state = 42): 
        self.train_dataset = DatasetReader(csv_file=train_file)
        self.test_dataset = DatasetReader(csv_file=test_file)

        self.X_train = self.train_dataset.features
        self.y_train = self.train_dataset.target
        
        self.X_test = self.test_dataset.features
        self.y_test = self.test_dataset.target
        
        self.model = IsolationForest(
            n_estimators=n_estimators, 
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1 
        )
        
    def train(self):
        self.model.fit(self.X_train)
        
    def test(self):
        self.train_preds = self.model.predict(self.X_train)
        self.test_preds = self.model.predict(self.X_test)

        self.train_preds = np.where(self.train_preds == 1, 0, 1)  
        self.test_preds = np.where(self.test_preds == 1, 0, 1)  
        
        self.train_accuracy = accuracy_score(self.y_train, self.train_preds)
        self.test_accuracy = accuracy_score(self.y_test, self.test_preds)

        print(f"Training Accuracy: {self.train_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {self.test_accuracy * 100:.2f}%")
        
    def commenceTraining(self):
        self.train()
        self.test()
        
        
def main():
    train_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/balanced.csv"
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/creditcard.csv"
    
    print("_________________________________________________________________________")
    print(" Neural Network using logisstic regression")
    model = NeuralNetworkModel(30, train_file, test_file)  
    model.commenceTraining()         
    
    print("_________________________________________________________________________")
    print("Logistic Regression")
    model = LogisticRegressionModel(30, train_file, test_file)
    model.commenceTraining()
    
    print("_________________________________________________________________________")
    print("XGBoost")
    model = XGBoostModel(train_file, test_file, 50, 0.005, 3, 73)
    model.commenceTraining()
    
    print("_________________________________________________________________________")
    print("Gradient Boosting Machine")
    model = GradientBoostingMachineModel(train_file, test_file)
    model.commenceTraining()
    
    print("_________________________________________________________________________")
    print("Random Forest")
    model = RandomForestModel(train_file, test_file) 
    model.commenceTraining()
    
    print("_________________________________________________________________________")
    print("Isolated Forest")
    train_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/creditcard.csv"
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/balanced.csv"
    model = IsolationForestModel(train_file, test_file)
    model.commenceTraining()

main()