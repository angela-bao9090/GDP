from SuperUseModel import UseModelPyTorch as PTM , UseModelSklearn as SKL






class UseModel:
    '''
    Class for handling and switching between different ML models (PyTorch or Sklearn) based on file type.



    Attributes:
        model - The loaded model (could be either PyTorch or Sklearn model)  - object
       
       
       
    Methods:
        __init__ - Initializes the class by loading a model file. If the provided model file is invalid, prompts the user to enter a valid one.
        
            Parameters:
                model_file - Path to the model file (can be .pth for PyTorch or .joblib for Sklearn) - str

        ChangeModel - Loads a model based on the provided file type. If the model file is a PyTorch (.pth) file, it loads a PyTorch model. If it's a Sklearn (.joblib) file, it loads an Sklearn model.
        
            Parameters:
                model_file - Path to the model file (can be .pth or .joblib) - str
    '''
    
    def __init__(self, model_file):
        self.model = None
        self.ChangeModel(model_file)
        while self.model == None:
            model_file = input("Enter a different model file: ")
            self.ChangeModel(model_file)
        
            
    def ChangeModel(self, model_file):
        if model_file[-4:] == ".pth":
            self.model = PTM(model_file)
        elif model_file[-7:] == ".joblib":
            self.model = SKL(model_file)
        else:
            print("Error: Invalid model file")






def main():
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTest.csv"
    current = UseModel("LRModel_20250328_180808.pth")
    
    #model.setThreshold(0.76)
    print(f"threshold: {current.model.threshold}")
    current.model.loadDatabaseFile(test_file)
    current.model.testOnFile()
    current.model.testOnFile()
    current.model.predictOnFile()
    current.model.testOnFile()
    current.model.loadInput([[0]*11, [1]*11])
    current.model.testOnInput()
    current.model.predictOnInput()
    current.model.testOnInput()
    
    current.ChangeModel("LRModel_20250328_221221.joblib")
    current.model.loadDatabaseFile(test_file)
    current.model.testOnFile()
    current.model.testOnFile()
    current.model.predictOnFile()
    current.model.testOnFile()
    current.model.loadInput([[0]*11, [1]*11])
    current.model.testOnInput()
    current.model.predictOnInput()
    current.model.testOnInput()
    
    
    
    
    
    
if __name__ == "__main__":
    main()