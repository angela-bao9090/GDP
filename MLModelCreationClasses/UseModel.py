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






def main(): # Example code
    test_file = "/Users/connorallan/Desktop/DOJO_project/ML/DataSets/fraudTest.csv"
    current = UseModel("CatBoostModel_20250411_155053.joblib")
    
    current.model.loadDatabaseFile(test_file)
    current.model.testOnFile()
    
    current.model.resetStored()
    
    #current.model.setThreshold(0.999)
    current.model.testOnFile()
    current.model.testOnFile() 
    current.model.resetStored()
    
    current.model.loadInput([[1857, 3560725013359375, 842.65, 79759,31.8599, -102.7413, 23, 1371855736, 31.315782000000000, -102.73639, 1]])
    current.model.testOnInput()
    current.model.loadInput([[13876,3593118134380341,4.64,4680,44.4971,-67.9503,1131,1372203439,44.031055,-68.81560400000000,1]])
    current.model.testOnInput()
    current.model.loadInput([[2506,3524574586339330,777.45,32960,27.633000000000000,-80.4031,105638,1371873336,27.913215,-80.423565,1]])
    current.model.testOnInput()
    
    #model.setThreshold(0.76)
    print(f"threshold: {current.model.threshold}")
    current.model.loadDatabaseFile(test_file)
    current.model.testOnFile()
    current.model.testOnFile()
    current.model.predictOnFile()
    #current.model.setThreshold(0.999)
    current.model.testOnFile()
    current.model.loadInput([[0]*11, [1]*11])
    current.model.testOnInput()
    current.model.predictOnInput()
    current.model.testOnInput()
    
    current.ChangeModel("RFModel_20250411_155056.joblib")
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