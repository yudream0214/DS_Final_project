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
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve

# subcode input
#from cluster_method import Cluster_Method
from DataTransforms import Data_Transfroms
from encoding import Encoding_Method
from Classifier import CLF_Method
from imbalance import Balance_Method

from collections import Counter

import argparse
import sys
import itertools
import seaborn as sns

# Data_Split (train & test)
class Data_Split():
    def __init__(self, df):
        
        PM21_X = df.iloc[1:, 3:-1]
        PM21_y = df.iloc[1:, -1]
    
        self.PM21_data_x = PM21_X
        self.PM21_data_y = PM21_y
        
        self.data_x = df.iloc[:, :-1]
        self.data_y = df.iloc[:, -1]

        
    def Split_Set(self, split_rate, random_num):
        X_train, X_test, y_train, y_test = train_test_split(self.PM21_data_x, self.PM21_data_y, test_size=split_rate, random_state=random_num, stratify=self.PM21_data_y)
        return X_train, X_test, y_train, y_test
    
    def Normal_Split_Set(self, split_rate, random_num):
        X_train, X_test, y_train, y_test = train_test_split(self.data_x, self.data_y, test_size=split_rate, random_state=random_num, stratify=self.data_y)
        return X_train, X_test, y_train, y_test
    
    
    # v3 切法 依照整筆資料去切
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
    df.drop_duplicates(inplace=True)
    
    # csv 
    # data_path = "spambase.csv"
    # df = pd.read_csv(data_path)
    
    print("org shape : ", df.shape)

    def drop_null(df):
        df_drop = df.dropna(axis=1,how='any') #drop columns that have any NaN values
        return df_drop
    
    df_droped = drop_null(df)
    cols = df_droped.columns
    
    print("null num :", df_droped.isnull().sum().sum())
    print(df_droped.shape)
    #df_droped.drop_duplicates(inplace=True)


    # Data Split
    split_rate = 0.2 
    random_num = 12
    
    DataSplit = Data_Split(df_droped)
    
    #X_train, X_test, y_train, y_test = DataSplit.Time_Split_v4(df_droped) #(df_droped)
    
    #X_train, X_test, y_train, y_test = DataSplit.Split_Set(split_rate, random_num)

    X_train, X_test, y_train, y_test = DataSplit.Time_Split(df_droped)
    
    #X_train, X_test, y_train, y_test = DataSplit.Normal_Split_Set(split_rate, random_num)


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
    
    # y Encoding
    # train_Encoding = Encoding_Method(y_train_df.astype(int))
    # y_train_onehot = train_Encoding.Onehot_Encoding() 
    
    # test_Encoding = Encoding_Method(y_test.astype(int))
    # y_test_onehot = test_Encoding.Onehot_Encoding()
    bb = x_train_df.iloc[:,:5]
    dd = y_train_df.iloc[:]
    aa = pd.concat([bb, dd], axis=1)
    sns_plot = sns.pairplot(aa[:1000], hue='class')
    sns_plot.figure.savefig("pairplot.png")  
    
    
    # imbalance 
    imb_neighbors = 3
    imb_random = 12
    
    # label encoding
    Imbalance = Balance_Method(x_train_df, y_train_df)
    

    # undersample
    #X_cnn, y_cnn = Imbalance.CNN(imb_neighbors, imb_random)
    #X_nm, y_nm = Imbalance.NM(imb_neighbors)
    #X_cc, y_cc = Imbalance.CC(imb_neighbors)
    X_enn, y_enn = Imbalance.Edited_NN(imb_neighbors)
    #X_ncr, y_ncr = Imbalance.NCR(imb_neighbors)   
    #X_tl, y_tl = Imbalance.TL(imb_neighbors)   
    #X_oss, y_oss = Imbalance.OSS(imb_neighbors, imb_random)   

    # oversample
    #X_smote, y_smote = Imbalance.SMOTE(imb_neighbors, imb_random)    
    #X_ada, y_ada = Imbalance.ADASYN(imb_neighbors, imb_random)  
    
    X_train_balanced , y_train_balanced =  X_enn, y_enn  
    
    # Classification  X_train, X_test, y_train, y_test
    
    
    # label encoding
    CLF = CLF_Method(X_train_balanced, x_test_df , y_train_balanced, y_test_df, 12)
    
    
    Forest_clf = CLF.RandomForest_clf()
    Xgb_clf = CLF.XGBboost_clf()
    Mlp_clf = CLF.MLP_clf()
    Svm_clf = CLF.SVM_clf() # "LinearSVC", "NuSVC", default = SVC
    Lgb_clf = CLF.Lightgbm_clf(class_mode='binary', class_num = 2,) # class_mode = 'multiclass', 
                                                                    # class_num = int, 

    # Model Cross_Val 
    print("--------- K-Fold score ---------")
    
    
    x , y = X_train_balanced , y_train_balanced 
    
    cv_forest = Cross_Validation(Forest_clf, x, y)
    CV_forest = cv_forest.F1_KFold()
    
    cv_mlp = Cross_Validation(Mlp_clf, x, y)
    CV_mlp = cv_mlp.F1_KFold()
    
    cv_xgb = Cross_Validation(Xgb_clf, x, y)
    CV_xgb = cv_xgb.F1_KFold() 
    
    cv_svm = Cross_Validation(Svm_clf, x, y)
    CV_svm = cv_svm.F1_KFold() 
    
    cv_lgb = Cross_Validation(Lgb_clf, x, y)
    CV_lgb = cv_lgb.F1_KFold() 
    
    print("--------- Testing F1 score ---------")
    
    # label encoding
    f1_true, f1_pred = y_test_df , x_test_df
    
    
    # f1 score std
    Mlp_F1 = F1_score(Mlp_clf, f1_true, f1_pred)
    Mlp_score, Mlp_pred = Mlp_F1.Score()
    
    Forest_F1 = F1_score(Forest_clf, f1_true, f1_pred)
    Forest_score, Forest_pred = Forest_F1.Score()
    
    Xgb_F1 = F1_score(Xgb_clf, f1_true, f1_pred)
    Xgb_score, Xgb_pred = Xgb_F1.Score()
    
    Svm_F1 = F1_score(Svm_clf, f1_true, f1_pred)
    Svm_score, Svm_pred = Svm_F1.Score()
    
    Lgb_F1 = F1_score(Lgb_clf, f1_true, f1_pred)
    Lgb_score, Lgb_pred = Lgb_F1.Score()    
    
    
    plt.figure(figsize=(9,6))
    
    models = ["MLP", "Forest", "XGB", "SVM", "LGB"]  
    X = np.arange(len(models))
    width = 0.3
    
    Y1 = [ CV_mlp, CV_forest, CV_xgb, CV_svm, CV_lgb ]                  # Cross_Val_mean 
    Y2 = [ Mlp_score, Forest_score, Xgb_score, Svm_score, Lgb_score]    # F1 score

    plt.bar(X, Y1, alpha=0.9, width = 0.3, facecolor = 'lightskyblue', edgecolor = 'white', label='Cross_Val_mean', lw=1)
    plt.bar(X + width, Y2, alpha=0.9, width = 0.3, facecolor = 'yellowgreen', edgecolor = 'white', label='F1 Score', lw=1)
    
    plt.xticks(X + width / 2, models)
    y_ticks = np.arange(0, 1, 0.2)
    plt.yticks(y_ticks)
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.legend(loc = "lower right")  # upper left, center
    plt.savefig("Score_result.png")
    plt.show()
    
    def plot_confuse_data(true, pred, encode_mode):
        
        if encode_mode == "one-hot":    
            truelabel = np.array(true).argmax(axis=-1)   # 将one-hot转化为label
        else:
            truelabel = true
        
        classes = range(0,2)
        predictions = pred
        
        confusion = confusion_matrix(y_true=truelabel, y_pred=predictions)
        
        plt.imshow(confusion, cmap=plt.cm.Blues)
        indices = range(len(confusion))
        
        plt.xticks(indices, classes)
        plt.yticks(indices, classes)
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        plt.title('Confusion matrix')
    
        for first_index in range(len(confusion)):    
            for second_index in range(len(confusion[first_index])):
                plt.text(first_index, second_index, confusion[first_index][second_index])
       
        #plt.savefig("confusion_matrix.png")
        plt.show()


    chose_pred = Xgb_pred
    #plot_confuse_data(label_encoded_test_y, chose_pred, encode_mode="label")
    
    def plot_confuse_matrix(y_true, y_pred):
        target_names = ['0', '1']
        labels_names = [0, 1]
        df_class_report = classification_report(y_true, y_pred,labels=labels_names, 
                                                target_names=target_names )

        print(df_class_report)    
        
        #df = pd.DataFrame(df_class_report).transpose()
        #df.to_csv('classification_report.csv', index = False)
        
        cm = confusion_matrix(y_true, y_pred,labels=labels_names)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=target_names)
        disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
        plt.savefig("confusion_matrix_2.png")
        plt.show()
    
    # label encoding 
    plot_confuse_matrix(y_test_df, Xgb_pred)
    

    
    def report_to_df(report):
        report = [x.split(' ') for x in report.split('\n')]
        header = ['Class Name']+[x for x in report[0] if x!='']
        values = []
        for row in report[1:-5]:
            row = [value for value in row if value!='']
            if row!=[]:
                values.append(row)
        df = pd.DataFrame(data = values, columns = header)
        return df
    
    
    df_report_mlp= report_to_df(classification_report(y_test_df, Mlp_pred))
    df_report_svm = report_to_df(classification_report(y_test_df, Svm_pred))
    df_report_forest = report_to_df(classification_report(y_test_df, Forest_pred))
    df_report_xgb = report_to_df(classification_report(y_test_df, Xgb_pred))
    df_report_lgb = report_to_df(classification_report(y_test_df, Lgb_pred))

    df_report_mlp.to_csv('classification_report_mlp.csv', index = False)
    df_report_svm.to_csv('classification_report_svm.csv', index = False)
    df_report_forest.to_csv('classification_report_forest.csv', index = False)
    df_report_xgb.to_csv('classification_report_xgb.csv', index = False)
    df_report_lgb.to_csv('classification_report_lgb.csv', index = False)
    
    def plot_roc_curve(fper, tper):
        plt.plot(fper, tper, color='red', label='ROC')
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend()
        plt.savefig("ROC_curve.png")
        plt.show()
    
    
    
    prob = Xgb_clf.predict_proba(x_test_df)
    prob = prob[:, 1]
    fper, tper, thresholds = roc_curve(y_test_df, prob)
    plot_roc_curve(fper, tper)
    