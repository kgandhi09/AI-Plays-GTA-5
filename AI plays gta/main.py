from PIL import ImageGrab, Image
import cv2
import numpy as np
from directkeys import PressKey, ReleaseKey, W, A, S, D
import tensorflow as tf
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt

def conv_pred_to_world(world, pred, world_height, world_width):
    #cv2.imshow('pred', pred)
    black_img = np.zeros(world.shape, np.uint8)
    white_pixels_pred = np.argwhere(pred >= 0.05)
    # (x,y) of pred world are : (white_pixels_pred[pixel][1],white_pixels_pred[pixel][0])
    for pixel in range(len(white_pixels_pred)):
        x_world = (white_pixels_pred[pixel][1]*world_height)/128
        y_world = (white_pixels_pred[pixel][0]*world_width)/128
        cv2.circle(world, (int(x_world),int(y_world)), 2, (0, 0, 255), 2)
    cv2.imshow('world', world)

def run(model):
    WORLD_HEIGHT = 1280
    WORLD_WIDTH = 1024
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    while True:
        img = np.array(ImageGrab.grab(bbox=(0,40,WORLD_HEIGHT,WORLD_WIDTH)))       #Window read
        img_resized = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))        
        X_test[0] = img_resized
        pred_val = model.predict(X_test)
        
        conv_pred_to_world(img, pred_val[0], WORLD_HEIGHT, WORLD_WIDTH)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    
    model = tf.keras.models.load_model('gta_lane_model.h5')
    run(model)
