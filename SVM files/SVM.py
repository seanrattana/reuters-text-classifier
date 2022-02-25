from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from Processor import processDF
import numpy as np
import pandas as pd
import joblib

def createSVM(dfToTrain, svmName):
    vectorizer = TfidfVectorizer(max_features=5000)
    SVM = svm.SVC(C=1.0, kernel='linear')
    tfidf_svm = Pipeline([('tfidf', vectorizer), ('svc', SVM)])
    
    Train_Y = dfToTrain["topics"].astype("int")
    
    tfidf_svm.fit(dfToTrain["processedData"], Train_Y)
    filename = svmName + ".pkl"
    joblib.dump(tfidf_svm, filename) 
    

def runSVM(svmName, testData):
    filename = svmName + ".pkl"
    loadSVM = joblib.load(filename)
    Test_Y = testData["topics"].astype("int")
    predictions_SVM = loadSVM.predict(testData["processedData"])
    print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)

df_train = processDF(pd.read_csv("./apteTrain.csv"))
createSVM(df_train, "pipeline")

df_test = processDF(pd.read_csv("./apteTest.csv"))
runSVM("pipeline", df_test)
