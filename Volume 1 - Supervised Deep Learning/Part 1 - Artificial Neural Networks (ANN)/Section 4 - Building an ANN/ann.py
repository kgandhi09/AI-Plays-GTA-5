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
x = dataset.iloc[:, 3:13].values    #input
y = dataset.iloc[:, 13].values      #output

#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le_x1 = LabelEncoder()
x[:, 1] = le_x1.fit_transform(x[:, 1])
le_x2 = LabelEncoder()
x[:, 2] = le_x2.fit_transform(x[:, 2])

ohc = OneHotEncoder(categorical_features = [1])
x = ohc.fit_transform(x).toarray()
x = x[:, 1:]


#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

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
# To decide the number of nodes in the hidden layer - take the average of number of nodes in input an output layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

#Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Adding the final output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

#Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the training set
classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)
    
#-------------------------------------------------------------------------------------------------------

#Part 3 - Making the predictions and evaluating the model

#Predicting the test set results
y_pred = classifier.predict(X_test)
#converting predicted result from percent to boolean
y_pred = y_pred > 0.5


#Predict if the customer with the following informations will leave the bank or not:
"""
Geography - France
Credit Score - 600
Gender - Male
Age - 40
Tenure - 3
Bank Balance - 60000
Number of Products - 2
Has Credit Card - Yes
Is Active Member - Yes
Estimated Salary - 50000
"""

new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

new_prodiction = new_prediction > 0.5
    

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)









