# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:32 2022

@author: Jason
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier        
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.multiclass import OneVsRestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np


class CLF_Method():
    def __init__(self, X_train, X_test, y_train, y_test, random_num):
        print("CLF Model Chose")
        
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        self.random_num = random_num
        
        print(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape)
        print(self.X_train[:5], self.X_test[:5] , self.y_train[:5], self.y_test[:5])
        
    def RandomForest_clf(self):
        #print("Model RandomForestClassifier")
        clf = RandomForestClassifier(random_state=self.random_num, n_jobs=-1)
        Forest_clf = clf.fit(self.X_train, self.y_train)
        #score = Forest_clf.score(self.X_test, self.y_test)
        #print("RandomForest score :", score)
        return Forest_clf
        
    def XGBboost_clf(self):
        #print("Model XGBoost")
        xgb_clf = XGBClassifier()
        Xgb_clf = xgb_clf.fit(self.X_train, self.y_train)
        #score = Xgb_clf.score(self.X_test, self.y_test)
        #print("XGB score :", score)
        return Xgb_clf
        
    # multi => class_mode = 'multiclass', class_num = int, metric_set = "multi_error"
    # binary =>
     
    def Lightgbm_clf(self, class_mode, class_num):
        if class_mode == "multiclass":
            params = {'learning_rate': 0.01,
                      'objective': 'multiclass', 
                      'num_class': class_num, 
                      'metric': 'multi_logloss', }
            
            lgb_model = LGBMClassifier(**params, n_jobs=-1)
            lgb_clf=lgb_model.fit(self.X_train, self.y_train)    
            
            return lgb_clf
        
        
        else:
            params = {'learning_rate': 0.01, 
                      'objective': 'binary',
                      'metric': 'binary_logloss'} 
            
            lgb_model = LGBMClassifier(**params, n_jobs=-1)
            lgb_clf=lgb_model.fit(self.X_train, self.y_train) 
            
            return lgb_clf
        
    def MLP_clf(self):
        MLP_clf = MLPClassifier(solver='adam', alpha=1e-5, activation="relu", 
                             hidden_layer_sizes=(32, 16, 8, 8, 8), random_state=1)
        Mlp_clf = MLP_clf.fit(self.X_train, self.y_train)
        #score = Mlp_clf.score(self.X_test, self.y_test)
        #print('MLP score :', score)
        return Mlp_clf
    
    def SVM_clf(self, svm_mode="SVC"):
        if svm_mode == "LinearSVC":
            model = LinearSVC(random_state=0, tol=1e-5)
            LinearSVC_clf = model.fit(self.X_train, self.y_train)    
            #score = svm_clf.score(self.X_test, self.y_test)
            #print("Binary svm score :", score)
            return LinearSVC_clf
        elif svm_mode == "NuSVC":
            model = NuSVC()
            NuSVC_clf = model.fit(self.X_train, self.y_train)
            #score = svmm_clf.score(self.X_test, self.y_test)
            #print("Multi svm score :", score)
            return NuSVC_clf
        else:
            model = SVC(gamma='auto')
            SVC_clf = model.fit(self.X_train, self.y_train)
            #score = svmm_clf.score(self.X_test, self.y_test)
            #print("Multi svm score :", score)
            return SVC_clf
        