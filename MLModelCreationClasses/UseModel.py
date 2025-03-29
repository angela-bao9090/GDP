from SuperUseModel import UseModelPyTorch as PTM , UseModelSklearn as SKL

class UseModel:
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