# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:46:51 2022

@author: Jason
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.cluster import KMeans

# subcode input
from cluster_method import Cluster_Method
from DataTransforms import Data_Transfroms
from encoding import Encoding_Method
from Classifier import CLF_Method
#from imbalance import Balance_Method

from collections import Counter

import argparse
import sys
import itertools
import seaborn as sns




# Data_Split (train & test)
class Data_Split():
    def __init__(self, df):
        
        X = df.iloc[1:, 3:-1]
        y = df.iloc[1:, -1]
    
        self.data_x = X
        self.data_y = y
                
    def Split_Set(self, split_rate, random_num):
        X_train, X_test, y_train, y_test = train_test_split(self.data_x, self.data_y, 
                                                            test_size=split_rate, 
                                                            random_state=random_num, 
                                                            stratify=self.data_y)
        return X_train, X_test, y_train, y_test
    
    def Time_Split(self, df):
        print(" === 時間序列切割 === ")
        # train 
        X_train_1 = df.iloc[1:3075, 3:-1]
        X_train_2 = df.iloc[4707:, 3:-1]
        
        y_train_1 = df.iloc[1:3075, -1]
        y_train_2 = df.iloc[4707:, -1]
        
        X_train = pd.concat([X_train_1, X_train_2 ], axis=0, ignore_index=True)
        y_train = pd.concat([y_train_1, y_train_2], axis=0, ignore_index=True)
        
        # test
        X_test = df.iloc[3075:4707, 3:-1]
        y_test = df.iloc[3075:4707, -1]
        
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        return X_train, X_test, y_train, y_test
    
    # v4 切法 依照整筆資料去切
    def Time_Split_v4(self, df):
        print(" === 時間序列切割 === ")
        # train 
        X_train_1 = df.iloc[1:208, 3:-1]      # 0
        X_train_2 = df.iloc[261:430:, 3:-1]   # 1
        X_train_3 = df.iloc[473:641, 3:-1]    # 0
        X_train_4 = df.iloc[683:711:, 3:-1]   # 1
        X_train_5 = df.iloc[719:1857, 3:-1]   # 0
        X_train_6 = df.iloc[2391:2954, 3:-1]  # 1
        X_train_7 = df.iloc[3095:3434, 3:-1]  # 0
        X_train_8 = df.iloc[3518:3826, 3:-1]  # 1
        X_train_9 = df.iloc[3903:7669, 3:-1]  # 0

        y_train_1 = df.iloc[1:208, -1]      # 0
        y_train_2 = df.iloc[261:430:, -1]   # 1
        y_train_3 = df.iloc[473:641,  -1]   # 0
        y_train_4 = df.iloc[683:711:, -1]   # 1
        y_train_5 = df.iloc[719:1857, -1]   # 0
        y_train_6 = df.iloc[2391:2954, -1]  # 1
        y_train_7 = df.iloc[3095:3434, -1]  # 0
        y_train_8 = df.iloc[3518:3826, -1]  # 1
        y_train_9 = df.iloc[3903:7669, -1]  # 0

        X_train = pd.concat( [X_train_1, X_train_2, X_train_3, X_train_4, X_train_5, X_train_6, X_train_7, X_train_8 ,X_train_9], axis=0, ignore_index=True) 
        y_train = pd.concat( [y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8 ,y_train_9], axis=0, ignore_index=True) 

        # test
        X_test_1 = df.iloc[208:261, 3:-1]    # 0
        X_test_2 = df.iloc[430:473, 3:-1]   # 1
        X_test_3 = df.iloc[641:683, 3:-1]    # 0
        X_test_4 = df.iloc[711:719, 3:-1]   # 1
        X_test_5 = df.iloc[1857:2391, 3:-1]   # 0
        X_test_6 = df.iloc[2954:3095, 3:-1]  # 1
        X_test_7 = df.iloc[3434:3518, 3:-1]  # 0
        X_test_8 = df.iloc[3826:3903, 3:-1]  # 1
        X_test_9 = df.iloc[7669:, 3:-1]  # 0

        y_test_1 = df.iloc[208:261, -1]    # 0
        y_test_2 = df.iloc[430:473, -1]   # 1
        y_test_3 = df.iloc[641:683, -1]    # 0
        y_test_4 = df.iloc[711:719, -1]   # 1
        y_test_5 = df.iloc[1857:2391, -1]   # 0
        y_test_6 = df.iloc[2954:3095, -1]  # 1
        y_test_7 = df.iloc[3434:3518, -1]  # 0
        y_test_8 = df.iloc[3826:3903, -1]  # 1
        y_test_9 = df.iloc[7669:, -1]  # 0
        
        X_test = pd.concat( [X_test_1, X_test_2, X_test_3, X_test_4, X_test_5, X_test_6, X_test_7, X_test_8 ,X_test_9], axis=0, ignore_index=True) 
        y_test = pd.concat( [y_test_1, y_test_2, y_test_3, y_test_4, y_test_5, y_test_6, y_test_7, y_test_8 ,y_test_9], axis=0, ignore_index=True) 

        return X_train, X_test, y_train, y_test
    
    
# Cross Validation    
class Cross_Validation():
    def __init__(self, clf, X, y):
        
        self.clf_mode = clf
        self.X_crossv = X
        self.y_crossv = y 
        
    def F1_KFold(self):
        clf_name = str(self.clf_mode)
        clf_name = clf_name.split("(")[0]
        
        score = cross_val_score(self.clf_mode,self.X_crossv, self.y_crossv,cv=5, scoring = 'f1_micro').mean()
        
        print( clf_name + "_k_fold" , score)
        return score
    
    def Acc_KFold(self):
        print(cross_val_score(self.clf_mode, self.X_crossv, self.y_crossv, cv=5, scoring = 'accuracy').mean())


# F1_score
class F1_score():
    def __init__(self, clf, true, test):
        
        self.true = true
        self.clf = clf
        self.pred = clf.predict(test)
        
    def Score(self):
        from sklearn.metrics import f1_score
        score = f1_score(self.true, self.pred, average='micro')
        clf_name = str(self.clf)
        clf_name = clf_name.split("(")[0]
        
        print( clf_name + "_f1_score :",  score)
        
        return score, self.pred

if __name__ == '__main__':

    data_path = "Cycle_drugs_yu.xlsx"
    df = pd.read_excel(data_path)
    print("org shape : ", df.shape)

    def drop_null(df):
        df_drop = df.dropna(axis=1,how='any') #drop columns that have any NaN values
        return df_drop
    
    df_droped = drop_null(df)
    cols = df_droped.columns
    
    print("null num :", df_droped.isnull().sum().sum())
    print(df_droped.shape)

    #X = df_droped.iloc[:, 3:-1]
    #y = df_droped.iloc[:, -1]
    #print(Counter(y))

    # Data Split
    split_rate = 0.2 
    random_num = 12
    
    DataSplit = Data_Split(df_droped)
    X_train, X_test, y_train, y_test = DataSplit.Time_Split_v4(df_droped)
    
    #X_train, X_test, y_train, y_test = DataSplit.Split_Set(split_rate, random_num)
    
    #X_train, X_test, y_train, y_test = DataSplit.Time_Split(df_droped)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    #print( " y_train :" , Counter(y_train),"\n", " y_test :" , Counter(y_test))
   
    
    # summarize distribution
    counter_train = Counter(y_train)
    for k1,v1 in counter_train.items():
        per1 = v1 / len(y_train) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k1, v1, per1))   
    
    counter_test = Counter(y_test)
    for k2,v2 in counter_test.items():
        per2 = v2 / len(y_test) * 100
        print('Class=%d, n=%d (%.3f%%)' % (k2, v2, per2)) 
    

    # Data_Transfrom  
    Method_Transforms = Data_Transfroms(X_train, X_test)
    X_train_scale, X_test_scale = Method_Transforms.MinMax_Scaler()
    
    # train 
    x_train_df = pd.DataFrame(X_train_scale)
    y_train_df = pd.Series(y_train, )
    y_train_df.reset_index(drop=True, inplace=True)
    
    # test
    x_test_df = pd.DataFrame(X_test_scale)
    y_test_df = pd.Series(y_test, )
    y_test_df.reset_index(drop=True, inplace=True)
    
    
    print(" === train  data === ")
    print(x_train_df.shape)
    print(y_train_df.shape)
    y_train_df = y_train_df.astype(int)
    
    print(" === test data === " )
    print(x_test_df.shape)
    print(y_test_df.shape)   
    y_test_df = y_test_df.astype(int)

    print(" === === === === ===")
        
    # clustering 
    Clustering = Cluster_Method(x_train_df, y_train_df)
    
    cl_neighbors = 3
    cl_random = 12

    #X_km_0, y_km_0, X_km_1, y_km_1, X_km_2, y_km_2 = Clustering.Cluster_KMeans(cl_neighbors , cl_random)  # (x, y) 3 組 
    #X_minkm_0, y_minkm_0, X_minkm_1, y_minkm_1, X_minkm_2, y_minkm_2, = Clustering.Cluster_MiniBatchKMeans(cl_neighbors, cl_random)
    #X_cc, y_cc = Clustering.Cluster_DBSCAN(cl_neighbors)
    X_birch_0, y_birch_0, X_birch_1, y_birch_1, X_birch_2, y_birch_2 = Clustering.Cluster_Birch(cl_neighbors)
    #X_ncr_0, y_ncr_0, X_ncr_1, y_ncr_1, X_ncr_2, y_ncr_2 = Clustering.Cluster_SpectralClustering(cl_neighbors, cl_random)   

    #X_cluster_0, y_cluster_0, X_cluster_1, y_cluster_1, X_cluster_2, y_cluster_2 = X_birch_0, y_birch_0, X_birch_1, y_birch_1, X_birch_2, y_birch_2
    #X_cluster_0, y_cluster_0, X_cluster_1, y_cluster_1, X_cluster_2, y_cluster_2 = X_ncr_0, y_ncr_0, X_ncr_1, y_ncr_1, X_ncr_2, y_ncr_2
    X_cluster_0, y_cluster_0, X_cluster_1, y_cluster_1, X_cluster_2, y_cluster_2 = X_birch_0, y_birch_0, X_birch_1, y_birch_1, X_birch_2, y_birch_2

    print(X_cluster_0.shape, y_cluster_0.shape," / ", X_cluster_1.shape, y_cluster_1.shape, " / ", X_cluster_2.shape, y_cluster_2.shape)
    
    
    X_test_scale_df = pd.DataFrame(X_test_scale)

    # test data save 
    with pd.ExcelWriter('test_df.xlsx') as writer:  
        X_test_scale_df.to_excel(writer, sheet_name='X_test_scale')
        y_test_df.to_excel(writer, sheet_name='y_test_scale')


    # cluster 0 save 
    with pd.ExcelWriter('cluster_0.xlsx') as writer0:  
        X_cluster_0.to_excel(writer0, sheet_name='X_cluster')
        y_cluster_0.to_excel(writer0, sheet_name='y_cluster')

    # cluster 1 save 
    with pd.ExcelWriter('cluster_1.xlsx') as writer1:  
        X_cluster_1.to_excel(writer1, sheet_name='X_cluster')
        y_cluster_1.to_excel(writer1, sheet_name='y_cluster')

    # cluster 2 save 
    with pd.ExcelWriter('cluster_2.xlsx') as writer2:  
        X_cluster_2.to_excel(writer2, sheet_name='X_cluster')
        y_cluster_2.to_excel(writer2, sheet_name='y_cluster')