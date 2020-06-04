#Convolutional Neural Network

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 

#part 1

#Initializing the CNN
classifier = Sequential()

#Step 1: Convolution (Adding convolution layers)
classifier.add(Convolution2D(32,3,3, input_shape=(64,64,3), activation = 'relu'))

#Step 2: Max Pooling (Reducing the size of the feature maps)
classifier.add(MaxPooling2D(pool_size = (2,2)))

#Step 3: Flattening (Taking all the pooled feature maps and putting them into one single column)
classifier.add(Flatten())        #This flattened layer is the input for ANN to further optimize the results

#Step 4: Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
#Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))     #Using signmoid function becuase only have two ouputs; if more than two outputs use softmax func


#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Part 2 
#Image Pre-Processing: Fitting the CNN to the images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('/home/kgandhi/Desktop/Deep_Learning/Dog_Cat_Image_Classification/cnn_dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('/home/kgandhi/Desktop/Deep_Learning/Dog_Cat_Image_Classification/cnn_dataset/test_set',
                                             target_size=(64, 64),
                                             batch_size=32,
                                             class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

#Part 3: Making a single prediction
test_image = image.load_img('/home/kgandhi/Desktop/Deep_Learning/Dog_Cat_Image_Classification/cnn_dataset/single_prediction/cat_or_dog_1.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
        prediction = 'Dog'
else:
        prediction = 'Cat'
        
print prediction