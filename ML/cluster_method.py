# -*- coding: utf-8 -*-
"""
Created on Sun May 22 22:01:21 2022

@author: Jason
"""

# -*- coding: utf-8 -*-
"""
Created on Sun May 15 19:06:33 2022

@author: Jason
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, confusion_matrix, multilabel_confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder

from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture

from category_encoders import TargetEncoder
from collections import Counter

import sys
import itertools
import seaborn as sns

class Cluster_Method():    
    def __init__(self, X, y):
      
        self.X_input = X
        self.y_input = y
    
    def Cluster_KMeans(self, n, random_num):
        print(" ==== KMeans ==== ")
        x = self.X_input
        y = self.y_input
        
        kmeans = KMeans(n_clusters = n, random_state = random_num)
        kmeans.fit(x)
        print(kmeans.labels_)
        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(kmeans.labels_).value_counts()
        print("cluster2聚类数量\n", (quantity))
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(kmeans.labels_)
        res0 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        print("类别为0的数据\n", x_train_0.shape, y_train_0.shape)
        
        y_train_0_series = y_train_0.iloc[:,0]  
        #print(y_train_0.describe())
        counter_clustered = Counter(y_train_0_series)
        for k3,v3 in counter_clustered.items():
            per3 = v3 / len(y_train_0_series) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k3, v3, per3)) 
        
        return x_train_0, y_train_0
        
        """
        plt.figure(figsize=(10,8))
        plt.rcParams['font.size'] = 14
        
        #以不同顏色畫出原始的 10 群資料
        plt.subplot(121)
        plt.title('Original data (10 groups)')
        plt.scatter(x.T[0], x.T[1], c=y, cmap=plt.cm.Set1)
        
        #根據重新分成的 5 組來畫出資料
        plt.subplot(122)
        plt.title('KMeans=5 groups')
        plt.scatter(x.T[0], x.T[1], c=kmeansed, cmap=plt.cm.Set1)
        
        #顯示圖表
        plt.tight_layout()
        plt.show()
        """

    def Cluster_MiniBatchKMeans(self, n, random_num):
        print(" ==== MiniBatchKMeans ==== ")
        
        x = self.X_input
        y = self.y_input
        
        MiniKMeans = MiniBatchKMeans(n_clusters = n, 
                                     random_state = random_num, 
                                     batch_size=4096) # MLK error 需依照PC memory設定 >= 4096
        MiniKMeans.fit(x)
        print(MiniKMeans.labels_)
        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(MiniKMeans.labels_).value_counts()
        print("cluster2聚类数量\n", (quantity))
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(MiniKMeans.labels_)
        res0 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        print("类别为0的数据\n", x_train_0.shape, y_train_0.shape)
        
        y_train_0_series = y_train_0.iloc[:,0]  
        #print(y_train_0.describe())
        counter_clustered = Counter(y_train_0_series)
        for k3,v3 in counter_clustered.items():
            per3 = v3 / len(y_train_0_series) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k3, v3, per3)) 
        
        return x_train_0, y_train_0
        
    def Cluster_DBSCAN(self, ep, sample):
        print(" ==== DBSCAN ====")
        
        x = self.X_input
        y = self.y_input    
        
        C_DBSCAN = DBSCAN(eps = ep, 
                          min_samples = sample ) # MLK error 需依照PC memory設定 >= 4096
        C_DBSCAN.fit(x)
        print(C_DBSCAN.labels_)
        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(C_DBSCAN.labels_).value_counts()
        print("cluster2聚类数量\n", (quantity))
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(C_DBSCAN.labels_)
        res0 = res0Series[res0Series.values == -1]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        print("类别为0的数据\n", x_train_0.shape, y_train_0.shape)
        
        y_train_0_series = y_train_0.iloc[:,0]  
        #print(y_train_0.describe())
        counter_clustered = Counter(y_train_0_series)
        for k3,v3 in counter_clustered.items():
            per3 = v3 / len(y_train_0_series) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k3, v3, per3)) 
        
        return x_train_0, y_train_0
        
    
    def Cluster_Birch(self, n):
        print(" ==== MiniBatchKMeans ==== ")
        
        x = self.X_input
        y = self.y_input
        
        C_Birch = Birch(n_clusters = n, )
        C_Birch.fit(x)
        print(C_Birch.labels_)
        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(C_Birch.labels_).value_counts()
        print("cluster2聚类数量\n", (quantity))
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(C_Birch.labels_)
        res0 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        print("类别为0的数据\n", x_train_0.shape, y_train_0.shape)
        
        y_train_0_series = y_train_0.iloc[:,0]  
        #print(y_train_0.describe())
        counter_clustered = Counter(y_train_0_series)
        for k3,v3 in counter_clustered.items():
            per3 = v3 / len(y_train_0_series) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k3, v3, per3)) 
        
        return x_train_0, y_train_0
    
    
    def Cluster_SpectralClustering(self, n, random_num):
        print(" ==== MiniBatchKMeans ==== ")
        
        x = self.X_input
        y = self.y_input
        
        C_SpectralClustering = SpectralClustering(n_clusters=n,
                                                  assign_labels='discretize',
                                                  random_state=random_num)
        C_SpectralClustering.fit(x)
        print(C_SpectralClustering.labels_)
        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(C_SpectralClustering.labels_).value_counts()
        print("cluster2聚类数量\n", (quantity))
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(C_SpectralClustering.labels_)
        res0 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        print("类别为0的数据\n", x_train_0.shape, y_train_0.shape)
        
        y_train_0_series = y_train_0.iloc[:,0]  
        #print(y_train_0.describe())
        counter_clustered = Counter(y_train_0_series)
        for k3,v3 in counter_clustered.items():
            per3 = v3 / len(y_train_0_series) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k3, v3, per3)) 
        
        return x_train_0, y_train_0

    def Cluster_GaussianMixture(self, n, random_num):
        print(" ==== MiniBatchKMeans ==== ")
        
        x = self.X_input
        y = self.y_input
        
        C_GaussianMixture = GaussianMixture(n_components = n , random_state=random_num)
        C_GaussianMixture.fit(x)
        print(C_GaussianMixture.means_ )
        
        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(C_GaussianMixture.means_ ).value_counts()
        print("cluster2聚类数量\n", (quantity))
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(C_GaussianMixture.means_ )
        res0 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        print("类别为0的数据\n", x_train_0.shape, y_train_0.shape)
        
        y_train_0_series = y_train_0.iloc[:,0]  
        #print(y_train_0.describe())
        counter_clustered = Counter(y_train_0_series)
        for k3,v3 in counter_clustered.items():
            per3 = v3 / len(y_train_0_series) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k3, v3, per3)) 
        
        return x_train_0, y_train_0
    
        
    


        