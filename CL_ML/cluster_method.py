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
from sklearn.decomposition import PCA

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
        
        def counter(y):
            counter_clustered = Counter(y)
            for k,v in counter_clustered.items():
                per = v / len(y) * 100
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per)) 
        
        x = self.X_input
        y = self.y_input
        
        kmeans = KMeans(n_clusters = n, random_state = random_num)
        kmeans.fit(x)
        print(kmeans.labels_)
        
        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(x)
        org_dis = pca_scale.transform(x) 

        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(kmeans.labels_).value_counts()
        print("cluster2聚类数量", "\n" , ( quantity ))
        print("------------------------------")
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(kmeans.labels_)
        
        res0 = res0Series[res0Series.values == 0]
        res1 = res0Series[res0Series.values == 1]
        res2 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        counter(y_train_0)
        print("Class 0 : ", x_train_0.shape, y_train_0.shape)
        print("------------------------------")
        
        x_train_1 = x.iloc[res1.index]
        y_train_1 = y.iloc[res1.index]
        counter(y_train_1)        
        print("Class 1 : ", x_train_1.shape, y_train_1.shape)
        print("------------------------------")
        
        x_train_2 = x.iloc[res2.index]
        y_train_2 = y.iloc[res2.index]
        counter(y_train_2)        
        print("Class 2 : ", x_train_2.shape, y_train_2.shape)
        print("------------------------------")


        pca_0 = pca_scale.transform(x_train_0) 
        pca_1 = pca_scale.transform(x_train_1) 
        pca_2 = pca_scale.transform(x_train_2) 

        # Cluster_KMeans_Distribution
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=y)
        
        plt.subplot(122)
        plt.title("KMeans Dis 0")
        sns.scatterplot(pca_0[:, 0], pca_0[:, 1], hue=y_train_0)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_KMeans_Distribution_1.png")
        
        plt.figure(figsize=(10,8))
        plt.subplot(121)
        plt.title("KMeans Dis 1")
        sns.scatterplot(pca_1[:, 0], pca_1[:, 1], hue=y_train_1)
        
        plt.subplot(122)
        plt.title("KMeans Dis 2")
        sns.scatterplot(pca_2[:, 0], pca_2[:, 1], hue=y_train_2)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_KMeans_Distribution_2.png")
        
        plt.show()


        return x_train_0, y_train_0, x_train_1, y_train_1, x_train_2, y_train_2
        

    def Cluster_MiniBatchKMeans(self, n, random_num):
        print(" ==== MiniBatchKMeans ==== ")
        
        x = self.X_input
        y = self.y_input
        
        def counter(y):
            counter_clustered = Counter(y)
            for k,v in counter_clustered.items():
                per = v / len(y) * 100
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per)) 
        
        MiniKMeans = MiniBatchKMeans(n_clusters = n, 
                                     random_state = random_num, 
                                     batch_size=4096) # MLK error 需依照PC memory設定 >= 4096
        MiniKMeans.fit(x)
        print(MiniKMeans.labels_)
        
        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(x)
        org_dis = pca_scale.transform(x) 

        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(MiniKMeans.labels_).value_counts()
        print("cluster2聚类数量", "\n" , ( quantity ))
        print("------------------------------")
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(MiniKMeans.labels_)
        
        res0 = res0Series[res0Series.values == 0]
        res1 = res0Series[res0Series.values == 1]
        res2 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        counter(y_train_0)
        print("Class 0 : ", x_train_0.shape, y_train_0.shape)
        print("------------------------------")
        
        x_train_1 = x.iloc[res1.index]
        y_train_1 = y.iloc[res1.index]
        counter(y_train_1)        
        print("Class 1 : ", x_train_1.shape, y_train_1.shape)
        print("------------------------------")
        
        x_train_2 = x.iloc[res2.index]
        y_train_2 = y.iloc[res2.index]
        counter(y_train_2)        
        print("Class 2 : ", x_train_2.shape, y_train_2.shape)
        print("------------------------------")


        pca_0 = pca_scale.transform(x_train_0) 
        pca_1 = pca_scale.transform(x_train_1) 
        pca_2 = pca_scale.transform(x_train_2) 

        # Cluster_KMeans_Distribution
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=y)
        
        plt.subplot(122)
        plt.title("MinKM Dis 0")
        sns.scatterplot(pca_0[:, 0], pca_0[:, 1], hue=y_train_0)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_MinKM_Distribution_1.png")
        
        plt.figure(figsize=(10,8))
        plt.subplot(121)
        plt.title("MinKM Dis 1")
        sns.scatterplot(pca_1[:, 0], pca_1[:, 1], hue=y_train_1)
        
        plt.subplot(122)
        plt.title("MinKM Dis 2")
        sns.scatterplot(pca_2[:, 0], pca_2[:, 1], hue=y_train_2)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_MinKM_Distribution_2.png")
        
        plt.show()

        return x_train_0, y_train_0, x_train_1, y_train_1, x_train_2, y_train_2
    
    
    def Cluster_SpectralClustering(self, n, random_num):
        print(" ==== MiniBatchKMeans ==== ")
        
        x = self.X_input
        y = self.y_input
        
        def counter(y):
            counter_clustered = Counter(y)
            for k,v in counter_clustered.items():
                per = v / len(y) * 100
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
        
        C_SpectralClustering = SpectralClustering(n_clusters=n,
                                                  assign_labels='discretize',
                                                  random_state=random_num)
        C_SpectralClustering.fit(x)
        print(C_SpectralClustering.labels_)
        
        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(x)
        org_dis = pca_scale.transform(x) 

        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(C_SpectralClustering.labels_).value_counts()
        print("cluster2聚类数量", "\n" , ( quantity ))
        print("------------------------------")
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(C_SpectralClustering.labels_)
        
        res0 = res0Series[res0Series.values == 0]
        res1 = res0Series[res0Series.values == 1]
        res2 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        counter(y_train_0)
        print("Class 0 : ", x_train_0.shape, y_train_0.shape)
        print("------------------------------")
        
        x_train_1 = x.iloc[res1.index]
        y_train_1 = y.iloc[res1.index]
        counter(y_train_1)        
        print("Class 1 : ", x_train_1.shape, y_train_1.shape)
        print("------------------------------")
        
        x_train_2 = x.iloc[res2.index]
        y_train_2 = y.iloc[res2.index]
        counter(y_train_2)        
        print("Class 2 : ", x_train_2.shape, y_train_2.shape)
        print("------------------------------")


        pca_0 = pca_scale.transform(x_train_0) 
        pca_1 = pca_scale.transform(x_train_1) 
        pca_2 = pca_scale.transform(x_train_2) 

        # Cluster_KMeans_Distribution
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=y)
        
        plt.subplot(122)
        plt.title("MinKM Dis 0")
        sns.scatterplot(pca_0[:, 0], pca_0[:, 1], hue=y_train_0)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_MinKM_Distribution_1.png")
        
        plt.figure(figsize=(10,8))
        plt.subplot(121)
        plt.title("MinKM Dis 1")
        sns.scatterplot(pca_1[:, 0], pca_1[:, 1], hue=y_train_1)
        
        plt.subplot(122)
        plt.title("MinKM Dis 2")
        sns.scatterplot(pca_2[:, 0], pca_2[:, 1], hue=y_train_2)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_MinKM_Distribution_2.png")
        
        plt.show() 

        return x_train_0, y_train_0, x_train_1, y_train_1, x_train_2, y_train_2
        
    def Cluster_DBSCAN(self, ep, sample):
        print(" ==== DBSCAN ====")
        
        x = self.X_input
        y = self.y_input    
        
        def counter(y):
            counter_clustered = Counter(y)
            for k,v in counter_clustered.items():
                per = v / len(y) * 100
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per)) 
        
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
        
        def counter(y):
            counter_clustered = Counter(y)
            for k,v in counter_clustered.items():
                per = v / len(y) * 100
                print('Class=%d, n=%d (%.3f%%)' % (k, v, per)) 
   
                
        C_Birch = Birch(n_clusters = n, )
        C_Birch.fit(x)
        print(C_Birch.labels_)
        
        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(x)
        org_dis = pca_scale.transform(x) 

        
        #print("clf_KMeans聚类中心\n", (clusted.cluster_centers_))
        quantity = pd.Series(C_Birch.labels_).value_counts()
        print("cluster2聚类数量", "\n" , ( quantity ))
        print("------------------------------")
        # 获取聚类之后每个聚类中心的数据
        res0Series = pd.Series(C_Birch.labels_)
        
        res0 = res0Series[res0Series.values == 0]
        res1 = res0Series[res0Series.values == 1]
        res2 = res0Series[res0Series.values == 2]
        
        x_train_0 = x.iloc[res0.index]
        y_train_0 = y.iloc[res0.index]
        counter(y_train_0)
        print("Class 0 : ", x_train_0.shape, y_train_0.shape)
        print("------------------------------")
        
        x_train_1 = x.iloc[res1.index]
        y_train_1 = y.iloc[res1.index]
        counter(y_train_1)        
        print("Class 1 : ", x_train_1.shape, y_train_1.shape)
        print("------------------------------")
        
        x_train_2 = x.iloc[res2.index]
        y_train_2 = y.iloc[res2.index]
        counter(y_train_2)        
        print("Class 2 : ", x_train_2.shape, y_train_2.shape)
        print("------------------------------")


        pca_0 = pca_scale.transform(x_train_0) 
        pca_1 = pca_scale.transform(x_train_1) 
        pca_2 = pca_scale.transform(x_train_2) 

        # Cluster_KMeans_Distribution
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=y)
        
        plt.subplot(122)
        plt.title("Birch Dis 0")
        sns.scatterplot(pca_0[:, 0], pca_0[:, 1], hue=y_train_0)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_Birch_Distribution_1.png")
        
        plt.figure(figsize=(10,8))
        plt.subplot(121)
        plt.title("Birch Dis 1")
        sns.scatterplot(pca_1[:, 0], pca_1[:, 1], hue=y_train_1)
        
        plt.subplot(122)
        plt.title("Birch Dis 2")
        sns.scatterplot(pca_2[:, 0], pca_2[:, 1], hue=y_train_2)

        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_Birch_Distribution_2.png")
        
        plt.show()

        return x_train_0, y_train_0, x_train_1, y_train_1, x_train_2, y_train_2
    


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
    
        
    


        