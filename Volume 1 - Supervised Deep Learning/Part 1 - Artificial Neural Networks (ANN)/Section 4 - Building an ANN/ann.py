# -*- coding: utf-8 -*-
"""
Spyder Editor


"""



#Part 1 - Data-preprocessing
#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing and slicing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:14].values    #input
y = dataset.iloc[:, 13].values      #output

#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x = LabelEncoder()
x[:, 1] = le_x.fit_transform(x[:, 1])
x[:, 2] = le_x.fit_transform(x[:, 2])

ohc = OneHotEncoder(categorical_features = [1])
x = ohc.fit_transform(x).toarray()
x = x[:, 1:]


#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_Train, Y_Test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature_Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#------------------------------------------------------------------------------------------------------

#Part 2 - Building Artificial Neural Network

#Importing the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the first input layer and first hidden layer
