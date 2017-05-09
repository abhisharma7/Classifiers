#!/home/hackpython/anaconda3/bin/python

from sklearn import cross_validation, neighbors
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
import pandas as pd



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
        pass

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
        pass
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
