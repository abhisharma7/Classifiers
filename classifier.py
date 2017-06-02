#!/home/hackpython/anaconda3/bin/python

# Author: Abhishek Sharma

from sklearn import cross_validation, neighbors,svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import KFold, cross_val_score
from sklearn.cross_validation import KFold
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import numpy as np
import pandas as pd
import sys

class classifier(object):

    def __init__(self,algorithm,dataset):
        self.iris_dataset = "dataset/irisdata150.txt"
        self.atntface_dataset = "dataset/ATNTFaceImages400.txt"
        self.atntletter_dataset = "dataset/HandWrittenLetters.txt"
        self.nba_dataset = "dataset/nba.xlsx"
        self.algorithm = algorithm
        self.dataset = dataset
        self.execute()

    def iris_dataset_preprocessing(self):
        
        df = pd.read_csv(self.iris_dataset,header=None)
        data = np.array(df.iloc[:,:4])
        label = np.array(df.iloc[:,4:])
        return data,label

    def atntface_dataset_preprocessing(self):

        df = pd.read_csv(self.atntface_dataset,header=None)
        df_train = df.drop(0,axis=0)
        train_list = []
        class_list = []
        for column in df_train.columns:
            nlist =[]
            clist = []
            nlist.append(df_train[column])
            train_list.append(nlist)
        X=np.array(train_list)
        dataset_size = len(X)
        data = X.reshape(dataset_size,-1)
        label = np.array(df.iloc[0])
        return data,label

    def atntletter_dataset_preprocessing(self):
        df = pd.read_csv(self.atntletter_dataset,header=None)
        df_train = df.drop(0,axis=0)
        train_list = []
        class_list = []
        for column in df_train.columns:
            nlist =[]
            clist = []
            nlist.append(df_train[column])
            train_list.append(nlist)
        X=np.array(train_list)
        dataset_size = len(X)
        data = X.reshape(dataset_size,-1)
        label =np.array(df.iloc[0])
        return data,label

    def nba_dataset_preprocessing(self):
        x1 = pd.ExcelFile(self.nba_dataset,header=None)
        df = x1.parse('Sheet1')
        label = np.array(df.iloc[:401,2:3])
        df = df.drop('Pos', 1)
        df = df.drop('Rk',1)
        df = df.drop('Player',1)
        df = df.drop('Tm',1)
        data = np.array(df.iloc[:401,0:10])
        test_data = np.array(df.iloc[402:476,0:10])
        return data, label, test_data

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
            data,label,test_data = self.nba_dataset_preprocessing()
            return data,label,test_data

    def knnclassifier(self): 
        data, label = self.dataset_preprocessing() 
        kf = KFold(n=len(data),n_folds=5,shuffle=True)
        for train_index, test_index in kf:
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            clf = neighbors.KNeighborsClassifier(n_neighbors=3)
            clf.fit(X_train,y_train)
            accuracy=clf.score(X_test,y_test)
            print(accuracy)

    def centriodclassifier(self):
        data,label = self.dataset_preprocessing()
        kf = KFold(n=len(data),n_folds=5,shuffle=True)
        for train_index,test_index in kf:
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            clf = NearestCentroid()
            clf.fit(X_train,y_train)
            accuracy=clf.score(X_test,y_test)
            print(accuracy)

    def linearclassifier(self):
        data,label = self.dataset_preprocessing()
        kf = KFold(n=len(data),n_folds=5,shuffle=True)
        for train_index,test_index in kf:
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            clf = LinearRegression(fit_intercept=True)
            clf.fit(X_train,y_train)
            accuracy = clf.score(X_test,y_test)
            print(accuracy)
    
    def svmclassifier(self):
        data,label = self.dataset_preprocessing()
        kf = KFold(n=len(data),n_folds=5,shuffle=True)
        for train_index,test_index in kf:
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            clf = svm.SVC(kernel='linear',gamma=2)
            clf.fit(X_train, y_train)
            accuracy = clf.score(X_test,  y_test)
            print(accuracy)
            
    def execute(self):

        if self.algorithm == "knn":
            self.knnclassifier()
        elif self.algorithm == "centroid":
            self.centriodclassifier()
        elif self.algorithm == "linear":
            self.linearclassifier()
        elif self.algorithm == "svm":
            self.svmclassifier()
        elif self.algorithm == "kmeans":
            self.kmeansclustering()
        else:
            sys.exit("algorithm not found")

if __name__ == '__main__':
    # Algorithms:
    # 1. K Nearest Neighbour    (knn)
    # 2. Centroid Classifier    (centroid)
    # 3. Linear Regression      (linear)
    # 4. Support Vector Machine (svm)
    # Dataset
    # 1. irisdataset
    # 2. atntface
    # 3. atntletter
    # 4. nba
    classifier("knn","atntletter")
    classifier("centroid","atntletter")
    classifier("linear","atntletter")
    classifier("svm","atntletter")
