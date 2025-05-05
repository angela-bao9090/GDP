import CreateModel as Models
import numpy as np
import matplotlib.pyplot as plt

train_file = r"C:\Users\maxhu\OneDrive\Documents\python code\gdp\archive\fraudTrain.csv"
test_file = r"C:\Users\maxhu\OneDrive\Documents\python code\gdp\archive\fraudTest.csv"

#This Python function, , calculates two key metrics precision and recall
#precision is accuracy but just for the actual fraud cases
#recall is the ones we got correct out of the ones we said where fraud 
def metrics(cm):
    [[tn, fp],[fn, tp]] = cm
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall

def f1Calc(precision,recall):
    p = precision
    r = recall
    if p + r == 0:
        return 0
    f = 2 * (p * r) / (p + r)
    return f

def plotGraph(lowerBound,upperBound, isLog, modelFunc, graphTitle, xAxisTitle):
    if(isLog):
        testVals = np.logspace(lowerBound,upperBound,10)
    else:
        testVals = np.arange(lowerBound, upperBound, (upperBound - lowerBound)/10)
    print(testVals)
    precision = [] #true positives/ predicted positives
    recall = [] #true positives/ all positives
    f1 = []
    for c in testVals:  
        model = modelFunc(c)
        model.train()
        model.test()
        p, r = metrics(model.cm)
        precision.append(p)
        recall.append(r)
        f = f1Calc(p,r)
        f1.append(f)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(testVals, precision, label="Precision", marker="o")
    plt.plot(testVals, recall, label="Recall", marker="s")
    plt.plot(testVals, f1, label= "f1", marker = "x")
    if(isLog):
        plt.xscale("log")  # Log scale for better visualization
    plt.xlabel(xAxisTitle)
    plt.ylabel("Score")
    plt.title(graphTitle)
    plt.legend()
    plt.grid(True)
    #plt.savefig("graphs/"+graphTitle+".png", dpi=300, bbox_inches='tight')
    plt.show()

def plotGraphInt(lowerBound,upperBound, modelFunc, graphTitle, xAxisTitle):
    testVals = np.round(np.arange(lowerBound, upperBound, (upperBound - lowerBound)/10)).astype(int)
    print(testVals)
    precision = [] #true positives/ predicted positives
    recall = [] #true positives/ all positives
    f1 = []
    for c in testVals:  
        model = modelFunc(c)
        model.train()
        model.test()
        p, r = metrics(model.cm)
        precision.append(p)
        recall.append(r)
        f = f1Calc(p,r)
        f1.append(f)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(testVals, precision, label="Precision", marker="o")
    plt.plot(testVals, recall, label="Recall", marker="s")
    plt.plot(testVals, f1, label= "f1", marker = "x")
    plt.xlabel(xAxisTitle)
    plt.ylabel("Score")
    plt.title(graphTitle)
    plt.legend()
    plt.grid(True)
    #plt.savefig("graphs/"+graphTitle+".png", dpi=300, bbox_inches='tight')
    plt.show()

def plotGraphTest(modelFunc,name):#to test the final choices for the model #takes in list of tuples names and models
        modelFunc.train()
        modelFunc.test()
        p, r = metrics(modelFunc.cm)
        f = f1Calc(p,r)
        print(name+": " + "precision: " +  "{:.2f}".format(p),"recall: " + "{:.2f}".format(r)+ " F1: " + "{:.2f}".format(f))
        return p,r,f

#graph for logistic regression
def genLogisticRegressionGraph():
    plotGraph(-4,-2,True,lambda c : Models.LogisticRegressionModel(train_file, test_file, C=c),
              "Logistic Regression - Regularisation", "Regularisation Parameter")
    plotGraph(0.6,0.9,False, lambda t: Models.LogisticRegressionModel(train_file, test_file,threshold = t),
              "Logistic Regression - Threshold", "Threshold ")
    plotGraph(100,300,False, lambda m: Models.LogisticRegressionModel(train_file, test_file,max_iter= m),
              "Logistic Regression - Max Iterations", "Max Iterations") # must take in int our graphing function comes up with floats
#new graphs 

def genNaivesBayes():
    plotGraph(0.8,1,False, lambda t: Models.NaiveBayes(train_file, test_file,threshold = t),
              "NaiveBayes - Threshold", "Threshold ")
    plotGraph(10^-6,10^-12,True, lambda v: Models.NaiveBayes(train_file, test_file,var_smoothing=v),
              "NaiveBayes - varSmoothing", "varSmoothing")

def genNeuralNetwork():
    plotGraph(0.6,0.9,False, lambda m: Models.NeuralNetwork(train_file, test_file, momentum=m),
              "NeuralNetwork - Momentum", "Momentum")
    plotGraph(0.95,1,False, lambda t: Models.NeuralNetwork(train_file, test_file,threshold = t),
              "NueralNetwork - Threshold", "Threshold ")
    plotGraph(10^-12,10^-2,True, lambda a: Models.NeuralNetwork(train_file, test_file,alpha= a),
              "NueralNetwork - alpha", "alpha")
def genRandomForest(): 
    '''plotGraph(0.5,1,False, lambda t: Models.RandomForestModel(train_file, test_file,threshold = t),
              "Random Forest - Threshold", "Threshold ") '''
    plotGraphInt(10,50, lambda d: Models.RandomForestModel(train_file, test_file,max_depth=d),
              "Random Forest - Depth", "Max Depth ")  
    plotGraphInt(40,70, lambda n: Models.RandomForestModel(train_file, test_file,n_estimators=n),
              "Random Forest - estimaters", "n-estimaters")
def genXGboost():
    plotGraph(0.8,1,False, lambda t: Models.XGBoostModel(train_file, test_file,threshold = t),
              "XGboost - Threshold", "Threshold ") 
    plotGraphInt(0,15, lambda d: Models.XGBoostModel(train_file, test_file,max_depth=d),
              "XGboost - Depth", "Max Depth ")  
    plotGraphInt(70,120, lambda n: Models.XGBoostModel(train_file, test_file,n_estimators=n),
              "XGboost - estimaters", "n-estimaters")
def genCatBoost(): 
    plotGraph(0.8,1,False, lambda t: Models.CatBoostModel(train_file, test_file,threshold = t),
              "CatBoost - Threshold", "Threshold ") 
    plotGraphInt(1,14, lambda d: Models.CatBoostModel(train_file, test_file,depth=d),
              "CatBoost- Depth", "Depth ") 
    plotGraphInt(100,1000, lambda i: Models.CatBoostModel(train_file, test_file,iterations=i),
              "Catboost - estimaters", "n-estimaters")
def genSGD():
    plotGraph(0.95,1,False, lambda t: Models.SGDClassifierModel(train_file, test_file,threshold = t),
              "SGD - Threshold", "Threshold ") 
    plotGraphInt(750,1250, lambda i: Models.SGDClassifierModel(train_file, test_file,max_iter=i),
              "SGD - iterations", "iterations")  

def testAllOptimal():
    precision = []
    recall = []
    names = []
    f1 = []
    #I couldn't loop this over a list a models are not listable
    #logistic regression
    p,r,f = plotGraphTest(Models.LogisticRegressionModel(train_file,test_file),"Logistic RegressionModel")
    precision.append(p)
    recall.append(r)
    f1.append(f)
    names.append("Logistic R")
    
    #naivebayes
    p,r,f = plotGraphTest(Models.NaiveBayes(train_file,test_file),"Naive Bayes")
    precision.append(p)
    recall.append(r)
    f1.append(f)
    names.append("NaivesBayes")
    
    #nueral network
    p,r,f = plotGraphTest(Models.NeuralNetwork(train_file,test_file),"Neural Network")
    precision.append(p)
    recall.append(r)
    f1.append(f)
    names.append("NeuralNetwork")

    #random Forest
    p,r,f = plotGraphTest(Models.RandomForestModel(train_file,test_file,max_depth=30, threshold = 0.7),"Random Forest")
    precision.append(p)
    recall.append(r)
    f1.append(f)
    names.append("Random Forest")

    #XGBoost 
    p,r,f = plotGraphTest(Models.XGBoostModel(train_file,test_file),"XGBoost")
    precision.append(p)
    recall.append(r)
    f1.append(f)
    names.append("XGBoost")

    #Stochastic Gradient Descent model
    p,r,f = plotGraphTest(Models.SGDClassifierModel(train_file,test_file),"SDG")
    precision.append(p)
    recall.append(r)
    f1.append(f)
    names.append("SDG")

    #CatBoost 
    p,r,f = plotGraphTest(Models.CatBoostModel(train_file,test_file),"Cat Boost")
    precision.append(p)
    recall.append(r)
    f1.append(f)
    names.append("Cat Boost")

    #ploting
    plotBar(names,precision,recall,f1)
    
def plotBar(names,precision,recall,f1):
    x = np.arange(len(names))
    #bar width
    barWidth = 0.35

    # Plotting the bars
    plt.bar(x - barWidth/3, precision, width=barWidth/3, label="Precision", color="blue")
    plt.bar(x , recall, width=barWidth/3, label="Recall", color="red")
    plt.bar(x + barWidth/3, f1, width = barWidth/3, label = "F1", color = "yellow")
    # Adding labels, title, and legend
    plt.xlabel("Names")
    plt.ylabel("Score")
    plt.title("Precision, Recall and F1")
    plt.xticks(x, names, rotation = 90)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()
    


def main():
    genLogisticRegressionGraph()
    genNaivesBayes()
    genNeuralNetwork()
    genRandomForest()
    genXGboost()
    genCatBoost()
    genSGD()
    testAllOptimal()


if __name__ == "__main__":
    main()
