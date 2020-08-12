import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow
import keras

#Part 1: Data Preprocessing
#Importing the dataset
dataset_train = pd.read_csv('/home/kgandhi/Desktop/Deep_Learning/stock_price_prediction_LSTM/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:,1:2].values

#Feature Scaling
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data stucture with 60 timesteps and 1 output
X_train = []
Y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i,0])
    Y_train.append(training_set_scaled[i,0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Building the RNN

#Initializing the RNN
regressor = Sequential()

#Making a robust stacked LSTM layer
#Adding the first LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#Adding the second LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#Adding the third LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#Adding the fourth LSTM layer and some Dropout Regularization
regressor.add(LSTM(units = 50, return_sequences = False))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Fitting the RNN to the Training set
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)
regressor.save('/home/kgandhi/Desktop/Deep_Learning/stock_price_prediction_LSTM/stock_prediction_rnn.h5')
#Making the predictions and visualizing the results

#Getting the real stock price of google
dataset_test = pd.read_csv('/home/kgandhi/Desktop/Deep_Learning/stock_price_prediction_LSTM/Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#Getting the predicted stock price
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []


for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])
    
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualizing the results
plt.plot(real_stock_price, color = 'green', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()