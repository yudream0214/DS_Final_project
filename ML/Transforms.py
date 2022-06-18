# -*- coding: utf-8 -*-
"""
Created on Mon May 23 00:53:07 2022

@author: Jason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder
from collections import Counter

import sys
import itertools
import seaborn as sns


class Data_Transfrom():
    def __init__(self, X_train, X_test): 
        
        self.X_train = X_train  # feature
        self.X_test = X_test  # target
        
        
        print("train test shape :", self.X_train.shape, self.X_test.shape)
        
    
    def Bin_Width(self, imped_data, bin_num):
        print("Binning Width")
            
        imped_data['bin_cols'] = pd.qcut(imped_data['class'], q=bin_num, duplicates='drop')
        
    def Bin_Depth(self, imped_data, bin_num):
        print("Binning Depth")
        
        #imped_data['bin_cols'] = pd.qcut(x, q)
        

    def Std_Scaler(self):
        print("Strand Scaler")
 
        std_scl = StandardScaler()
        std_scl = std_scl.fit(self.X_train)
        std_train_scled = std_scl.transform(self.X_train)
        std_test_scled = std_scl.transform(self.X_test)
        
        return std_train_scled, std_test_scled    

    def MinMax_Scaler(self):
        print("MinMax Scaler")
        
        MaxMin_scl = MinMaxScaler()
        MaxMin_scl = MaxMin_scl.fit(self.X_train)
        MaxMin_train_scl = MaxMin_scl.transform(self.X_train)
        MaxMin_test_scl = MaxMin_scl.transform(self.X_test)

        return MaxMin_train_scl, MaxMin_test_scl
    
    def Gernal_Scaler(self):
        print("Gernal scale")
    
        org_train = self.X_train
        org_test = self.X_test
        
        return org_train, org_test