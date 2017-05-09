#!/home/hackpython/anaconda3/bin/python

from sklearn import cross_validation, neighbors
#from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd
import sys

class classifier(object):

    def __init__(self,algorithm,dataset):
        self.iris_dataset = "dataset/irisdata150.txt"
        self.atntface_dataset = "dataset/ATNTFaceImages400.txt"
        self.atntletter_dataset = "dataset/HandWrittenLetters.txt"
        self.nba_dataset = "dataset/"
        self.algorithm = algorithm
        self.dataset = dataset
        self.execute()

    def iris_dataset_preprocessing(self):
        
        df = pd.read_csv(self.iris_dataset,header=None)
        data = np.array(df.iloc[:,:4])
        label = np.array(df.iloc[:,4:])
        return data,label

    def atntface_dataset_prepocessing(self):
        pass
    
    def atntletter_dataset_preprocessing(self):
        pass
    
    def nba_dataset_preprocessing(self):
        pass

    def dataset_preprocessing(self):
        
        if self.dataset == "irisdataset":
            data,label = self.iris_dataset_preprocessing()
            return data,label

        elif self.dataset == "atntface":
            data,label = self.atntface_dataset_preprocessing()
            return data,label

        elif self.dataset == "atntletter":
            data,label = self.atntletter_dataset_preprocessing()
            return data,label

        elif self.dataset == "nba":
            data,label = self.nba_dataset_preprocessing()
            return data,label

    def knnclassifier(self):
        
        data,label = self.dataset_preprocessing() 
        kf = KFold(n=len(data),n_folds=5,shuffle=True)
        for train_index, test_index in kf:
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            clf = neighbors.KNeighborsClassifier(n_neighbors=3)
            clf.fit(X_train,y_train)
            accuracy=clf.score(X_test,y_test)
            test = np.array([[6.1,2.8,4,1.3],[6.1,3,4.9,1.8],[5,3.4,1.5,0.2]])
            prediction = clf.predict(test)
            #print("TRAIN:", train_index, "TEST:", test_index)
            print(accuracy)
            print(prediction)

    def centriodclassifier(self):
        pass

    def linearclassifier(self):
        pass

    def svmclassifier(self):
        pass

    def execute(self):

        if self.algorithm == "knn":
            self.knnclassifier()
        elif self.algorithm == "centroid":
            pass
        elif self.algorithm == "linear":
            pass
        elif self.algorithm == "svm":
            pass
        elif self.algorithm == "kmeans":
            pass
        else:
            sys.exit("algorithm not found")

if __name__ == '__main__':
    # Algorithms:
    # 1. K Nearest Neighbour    (knn)
    # 2. Centroid Classifier    (centroid)
    # 3. Linear Regression      (linear)
    # 4. Support Vector Machine (svm)
    # 5. K-means Clustering     (kmeans)
    # Dataset
    # 1. irisdataset
    # 2. atntface
    # 3. atntletter
    # 4. nba
    classifier("knn","irisdataset")
