from PIL import ImageGrab
import cv2
import numpy as np
from directkeys import PressKey, ReleaseKey, W, A, S, D
import tensorflow as tf
from skimage.io import imread, imshow
from skimage.transform import resize


def conv_pred_to_world(world_array, pred_array, world_height, world_width):
    cv2.imshow('lane', pred_array)
    
    
def run(model):
    WORLD_HEIGHT = 1280
    WORLD_WIDTH = 1000
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    while True:
        img = np.array(ImageGrab.grab(bbox=(0,40,WORLD_HEIGHT,WORLD_WIDTH)))       #Window read
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))        
        X_test[0] = img
        pred_val = model.predict(X_test)
        
        conv_pred_to_world(img, pred_val[0], WORLD_HEIGHT, WORLD_WIDTH)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    
    model = tf.keras.models.load_model('gta_lane_model.h5')
    run(model)
