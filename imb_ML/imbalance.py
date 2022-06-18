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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from category_encoders import TargetEncoder
from collections import Counter

import sys
import itertools
import seaborn as sns
sns.set(style="white")

class Balance_Method():    
    def __init__(self, X, y):
        
        self.X_balance = np.array(X)
        self.y_balance = np.array(y)
        
    def CNN(self, neighbors_num, random_num):
        print("CondensedNearestNeighbour")
        from imblearn.under_sampling import CondensedNearestNeighbour 
        cnn = CondensedNearestNeighbour(n_neighbors=neighbors_num, random_state=random_num, n_jobs=-1) 
        X_CNN, y_CNN = cnn.fit_resample(self.X_balance, self.y_balance) 
            
            
        # summarize distribution
        counter = Counter(y_CNN)
        for k,v in counter.items():
            per = v / len(y_CNN) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

        # plot the distribution
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.title("CondensedNearestNeighbour Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter.keys(), counter.values())
        
        plt.subplot(212)
        plt.title("CondensedNearestNeighbour Balance Distribution")
        
        # scatter plot of examples by class label
        for label, _ in counter.items():
           	row_ix = np.where(y_CNN == label)[0]
           	plt.scatter(X_CNN[row_ix, 0], X_CNN[row_ix, 1], label=str(label))
        plt.legend()
        plt.tight_layout()
        plt.savefig("CondensedNearestNeighbour_Distribution.png")
        plt.show()

        return X_CNN, y_CNN 
        
    
    def NM(self, neighbors_num):
        print("NearMiss")        
        from imblearn.under_sampling import NearMiss 
        nm = NearMiss(n_neighbors=neighbors_num, n_jobs=-1)
        X_nm, y_nm = nm.fit_resample(self.X_balance, self.y_balance)
        
        # summarize distribution
        counter = Counter(y_nm)
        for k,v in counter.items():
            per = v / len(y_nm) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

        # plot the distribution
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.title("NearMiss Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter.keys(), counter.values())
        
        plt.subplot(212)
        plt.title("NearMiss Balance Distribution")
        
        # scatter plot of examples by class label
        for label, _ in counter.items():
           	row_ix = np.where(y_nm == label)[0]
           	plt.scatter(X_nm[row_ix, 0], X_nm[row_ix, 1], label=str(label))
        plt.legend()
        plt.tight_layout()
        plt.savefig("NearMiss_Distribution.png")
        plt.show()

        return X_nm, y_nm
        
    def CC(self, random_num):
        print("Cluster Centroids")
        from imblearn.under_sampling import ClusterCentroids  
        cc = ClusterCentroids(random_state=random_num, n_jobs=-1)
        X_cc, y_cc = cc.fit_resample(self.X_balance, self.y_balance)
        
        # summarize distribution
        counter = Counter(y_cc)
        for k,v in counter.items():
            per = v / len(y_cc) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))

        # plot the distribution
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.title("Cluster Centroids Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter.keys(), counter.values())
        
        plt.subplot(212)
        plt.title("Cluster Centroids Balance Distribution")
        
        # scatter plot of examples by class label
        for label, _ in counter.items():
           	row_ix = np.where(y_cc == label)[0]
           	plt.scatter(X_cc[row_ix, 0], X_cc[row_ix, 1], label=str(label))
        plt.legend()
        plt.tight_layout()
        plt.savefig("Cluster_Centroids_Distribution.png")
        plt.show()

        return X_cc, y_cc
    
    def Edited_NN(self, neighbors_num):
        print("EditedNearestNeighbours")
        from imblearn.under_sampling import EditedNearestNeighbours 
        enn = EditedNearestNeighbours(n_neighbors=neighbors_num)
        X_enn, y_enn = enn.fit_resample(self.X_balance, self.y_balance)
        
        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(self.X_balance)
        org_dis = pca_scale.fit_transform(self.X_balance) 
        sampled_dis = pca_scale.fit_transform(X_enn) 
        print("pca result :", org_dis[:5], sampled_dis[:5])

        # summarize distribution
        counter1 = Counter(self.y_balance)
        for k1,v1 in counter1.items():
            per1 = v1 / len(self.y_balance) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k1, v1, per1))
        
        
        counter2 = Counter(y_enn)
        for k2,v2 in counter2.items():
            per2 = v2 / len(y_enn) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k2, v2, per2))
        
        
        # bar plot
        
        plt.figure(figsize=(10,8))
        plt.tight_layout()

        plt.subplot(211)
        plt.title("Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter1.keys(), counter1.values())
        
        plt.subplot(212)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter2.keys(), counter2.values())
        plt.savefig("Data Distribution.png")

        # SMOTE Dis
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Imbalance Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=self.y_balance)
        
        plt.subplot(122)
        plt.title("ENN Balance Dis")
        sns.scatterplot(sampled_dis[:, 0], sampled_dis[:, 1], hue=y_enn)

        plt.legend()
        plt.tight_layout()
        plt.savefig("ENN Imblanece Distribution.png")
        plt.show()
        
        return X_enn, y_enn
        
    
    def NCR(self, neighbors_num):
        print("NeighbourhoodCleaningRule")
        from imblearn.under_sampling import NeighbourhoodCleaningRule 
        ncr = NeighbourhoodCleaningRule(n_neighbors=neighbors_num)
        X_ncr, y_ncr = ncr.fit_resample(self.X_balance, self.y_balance)
        
        # summarize distribution
        counter = Counter(y_ncr)
        for k,v in counter.items():
            per = v / len(y_ncr) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
       
        # plot the distribution
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.title("NeighbourhoodCleaningRule Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter.keys(), counter.values())
        
        plt.subplot(212)
        plt.title("NeighbourhoodCleaningRule Balance Distribution")
        
        # scatter plot of examples by class label
        for label, _ in counter.items():
           	row_ix = np.where(y_ncr == label)[0]
           	plt.scatter(X_ncr[row_ix, 0], X_ncr[row_ix, 1], label=str(label))
        plt.legend()
        plt.tight_layout()
        plt.savefig("NeighbourhoodCleaningRule_Distribution.png")
        plt.show()  
        
        return X_ncr, y_ncr
    
    
    def TL(self, neighbors_num):
        print("TomekLinks")
        from imblearn.under_sampling import TomekLinks 
        tl = TomekLinks()
        X_tl, y_tl = tl.fit_resample(self.X_balance, self.y_balance)
        
        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(self.X_balance)
        org_dis = pca_scale.fit_transform(self.X_balance) 
        sampled_dis = pca_scale.fit_transform(X_tl) 
        print("pca result :", org_dis[:5], sampled_dis[:5])

        # summarize distribution
        counter1 = Counter(self.y_balance)
        for k1,v1 in counter1.items():
            per1 = v1 / len(self.y_balance) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k1, v1, per1))
        
        
        counter2 = Counter(y_tl)
        for k2,v2 in counter2.items():
            per2 = v2 / len(y_tl) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k2, v2, per2))
        
        
        # bar plot
        
        plt.figure(figsize=(10,8))
        plt.tight_layout()

        plt.subplot(211)
        plt.title("Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter1.keys(), counter1.values())
        
        plt.subplot(212)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter2.keys(), counter2.values())
        plt.savefig("Data Distribution.png")

        # SMOTE Dis
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Imbalance Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=self.y_balance)
        
        plt.subplot(122)
        plt.title("TomekLinks Balance Dis")
        sns.scatterplot(sampled_dis[:, 0], sampled_dis[:, 1], hue=y_tl)

        plt.legend()
        plt.tight_layout()
        plt.savefig("TomekLinks Imblanece Distribution.png")
        plt.show()  
        
        return X_tl, y_tl
    
    def OSS(self, neighbors_num, random_num):
        print("OneSidedSelection")
        from imblearn.under_sampling import OneSidedSelection  
        oss = OneSidedSelection(n_neighbors=neighbors_num, random_state=random_num)
        X_oss, y_oss = oss.fit_resample(self.X_balance, self.y_balance)
        
        # summarize distribution
        counter = Counter(y_oss)
        for k,v in counter.items():
            per = v / len(y_oss) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
            
        # plot the distribution
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.title("OneSidedSelection Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter.keys(), counter.values())
        
        plt.subplot(212)
        plt.title("OneSidedSelection Balance Distribution")
        
        # scatter plot of examples by class label
        for label, _ in counter.items():
           	row_ix = np.where(y_oss == label)[0]
           	plt.scatter(X_oss[row_ix, 0], X_oss[row_ix, 1], label=str(label))
        plt.legend()
        plt.tight_layout()
        plt.savefig("OneSidedSelection_Distribution.png")
        plt.show()
        
        return X_oss, y_oss

    def SMOTE(self, neighbors_num, random_num):
        print("SMOTE")
        from imblearn.over_sampling import SMOTE 
        smote = SMOTE(k_neighbors=neighbors_num, random_state=random_num)
        X_smote, y_smote = smote.fit_resample(self.X_balance, self.y_balance)

        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(self.X_balance)
        org_dis = pca_scale.transform(self.X_balance) 
        sampled_dis = pca_scale.transform(X_smote) 
        print("pca result :", org_dis[:5], sampled_dis[:5])

        # summarize distribution
        counter1 = Counter(self.y_balance)
        for k1,v1 in counter1.items():
            per1 = v1 / len(self.y_balance) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k1, v1, per1))
        
        
        counter2 = Counter(y_smote)
        for k2,v2 in counter2.items():
            per2 = v2 / len(y_smote) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k2, v2, per2))
        
        
        # bar plot
        
        plt.figure(figsize=(10,8))
        plt.tight_layout()

        plt.subplot(211)
        plt.title("Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter1.keys(), counter1.values())
        
        plt.subplot(212)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter2.keys(), counter2.values())
        plt.savefig("Data Distribution.png")

        # SMOTE Dis
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Imbalance Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=self.y_balance)
        
        plt.subplot(122)
        plt.title("SMTOE Balance Dis")
        sns.scatterplot(sampled_dis[:, 0], sampled_dis[:, 1], hue=y_smote)

        plt.legend()
        plt.tight_layout()
        plt.savefig("SMTOE Imblanece Distribution.png")
        plt.show()
        
        return X_smote, y_smote

    def SVM_SMOTE(self, random_num):
        print("SVMSMOTE")
        from imblearn.over_sampling import SVMSMOTE 
        
        svm_smote = SVMSMOTE(k_neighbors=1, m_neighbors=10, random_state=random_num,)
        X_svm_smote, y_svm_smote = svm_smote.fit_resample(self.X_balance, self.y_balance)
        # summarize distribution
        counter = Counter(y_svm_smote)
        for k,v in counter.items():
            per = v / len(y_svm_smote) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
        # plot the distribution
        
        plt.figure(figsize=(10,8))
        plt.subplot(211)
        plt.title("SVMSMOTE Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter.keys(), counter.values())
        
        plt.subplot(212)
        plt.title("SVMSMOTE Balance Distribution")
        
        # scatter plot of examples by class label
        for label, _ in counter.items():
           	row_ix = np.where(y_svm_smote == label)[0]
           	plt.scatter(X_svm_smote[row_ix, 0], X_svm_smote[row_ix, 1], label=str(label))
        plt.legend()
        plt.tight_layout()
        plt.savefig("SVMSMOTE_Distribution.png")
        plt.show()
        
        return X_svm_smote, y_svm_smote
    
    def ADASYN(self, neighbors_num, random_num):
        print("ADASYN")
        from imblearn.over_sampling import ADASYN 
        
        ada = ADASYN(n_neighbors=neighbors_num, random_state=random_num)
        X_ada, y_ada = ada.fit_resample(self.X_balance, self.y_balance)

        # pca 
        print("PCA")
        pca_scale = PCA(n_components=2).fit(self.X_balance)
        org_dis = pca_scale.transform(self.X_balance) 
        sampled_dis = pca_scale.transform(X_ada) 
        print("pca result :", org_dis[:5], sampled_dis[:5])

        # summarize distribution
        counter1 = Counter(self.y_balance)
        for k1,v1 in counter1.items():
            per1 = v1 / len(self.y_balance) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k1, v1, per1))
        
        
        counter2 = Counter(y_ada)
        for k2,v2 in counter2.items():
            per2 = v2 / len(y_ada) * 100
            print('Class=%d, n=%d (%.3f%%)' % (k2, v2, per2))
        
        
        # bar plot
        
        plt.figure(figsize=(10,8))
        plt.tight_layout()

        plt.subplot(211)
        plt.title("Data Distribution")
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter1.keys(), counter1.values())
        
        plt.subplot(212)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.bar(counter2.keys(), counter2.values())
        plt.savefig("Data Distribution.png")

        # SMOTE Dis
        plt.figure(figsize=(10,8))
        
        plt.subplot(121)
        plt.title("Org Imbalance Dis")
        sns.scatterplot(org_dis[:, 0], org_dis[:, 1], hue=self.y_balance)
        
        plt.subplot(122)
        plt.title("ADASYN Balance Dis")
        sns.scatterplot(sampled_dis[:, 0], sampled_dis[:, 1], hue=y_ada)

        plt.legend()
        plt.tight_layout()
        plt.savefig("ADASYN Imblanece Distribution.png")
        plt.show()
    
        return X_ada, y_ada   
    
    # Ensemble Learning with Undersampling
    def EasyE(self, X, y, X_test, y_test, random_num):
        print("EasyEnsemble")
        from imblearn.ensemble import EasyEnsembleClassifier 
        from sklearn.metrics import confusion_matrix
        
        eec = EasyEnsembleClassifier(random_state=random_num)
        eec.fit(X, y)

        EasyEnsembleClassifier(...)
        y_pred = eec.predict(X_test, )
        print(confusion_matrix(y_test, y_pred))    

    def BCascade(self, X, y, X_test, y_test, random_num):
        print("BalanceCascade")
        from imblearn.ensemble import EasyEnsembleClassifier 
        from sklearn.metrics import confusion_matrix
        
        eec = EasyEnsembleClassifier(random_state=random_num)
        eec.fit(X, y)

        EasyEnsembleClassifier(...)
        y_pred = eec.predict(X_test, )
        print(confusion_matrix(y_test, y_pred)) 