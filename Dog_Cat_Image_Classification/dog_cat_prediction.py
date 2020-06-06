import keras
import numpy as np
from keras.preprocessing import image

#Part 3
#importing keras model
classifier = keras.models.load_model('/home/gandhi/Desktop/Deep_Learning/Dog_Cat_Image_Classification/dog_cat_cnn_model.h5')

#loading the test image
test_image = image.load_img('/home/gandhi/Desktop/Deep_Learning/Dog_Cat_Image_Classification/cnn_dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)

if result[0][0] == 1:
        prediction = 'Dog'
else:
        prediction = 'Cat'
        
print(prediction)