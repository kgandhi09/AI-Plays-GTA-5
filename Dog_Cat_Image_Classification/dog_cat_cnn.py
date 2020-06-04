#Convolutional Neural Network

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#part 1

#Initializing the CNN
classifier = Sequential()

#Step 1: Convolution (Adding convolution layers)
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation = 'relu'))