# -*- coding: utf-8 -*-


import csv
import random
import math
import operator
from statistics import mode
import pandas as pd
import numpy as np

class KNN :
    def __init__(self,train_set,test_set):
        self.train_set = train_set
        self.test_set = test_set
        
    
    def euclidean_distance(self,vec1, vec2):
        d=0
        for i in range(0,29):
            d += (vec1[i]-vec2[i])**2
        return d

    def get_neighbourhood(self,X_train, y_train, point, K):
        pairs = zip(X_train, y_train)
        pairs.sort(key = lambda pair: self.euclidean_distance(point, pair[0]))
        pairs = pairs[0:K]
        return mode(zip(*pairs)[1])
  
    def get_accuracy(self,pred, y_test):
        d=0
        for i in range(0,len(pred)):
            if pred[i] == y_test[i]:
                d += 1
        return float(d)/float(len(pred))
    
    def knn_main_code(self,X_train, X_test, y_train, K):
        pred=[]
        for x in X_test:
            pred.append(self.get_neighbourhood(X_train,y_train,x,K))
        return pred
