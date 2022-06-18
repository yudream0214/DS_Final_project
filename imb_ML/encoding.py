# -*- coding: utf-8 -*-
"""
Created on Sun May 15 18:53:01 2022

@author: Jason
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler,OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder

class Encoding_Method():
    def __init__(self, input_y):
        print("Encoding Method")
    
        #np_y = np.array(input_y)
        self.y = input_y
    
    def Label_Encoding(self):
        print("Labelhot mode")
        
        labelencoder = LabelEncoder()
        data_label_hot = labelencoder.fit_transform(self.y)
        #data_label_hot = data_label_hot.reshape(-1,1)
        return data_label_hot
        
    def Onehot_Encoding(self):
        print("Onehot mode")
        
        labelencoder = LabelEncoder()
        data_label_hot = labelencoder.fit_transform(self.y)
        data_label_hot = data_label_hot.reshape(-1,1)
        
        onehotencoder = OneHotEncoder(sparse=False)
        data_one_hot = onehotencoder.fit_transform(data_label_hot)
        #data_one_hot = data_one_hot.reshape(-1,1)
        return data_one_hot
    
    
    def Target_Encoding(self):
        print("Target mode")
        
        targetencoder = TargetEncoder()
        data_target_encode = targetencoder.fit_transform(self.y)
        data_target_encode = data_target_encode.reshape(-1,1)
        return data_target_encode