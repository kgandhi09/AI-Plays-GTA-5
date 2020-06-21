from PIL import ImageGrab
from win32 import win32gui
import cv2
import numpy as np
import time
from numpy import ones,vstack
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from statistics import mean
from directkeys import PressKey, ReleaseKey, W, A, S, D
import tensorflow as tf
from skimage.io import imread, imshow
from skimage.transform import resize

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_image(orig_image):
    #converting the original image to grayscale
    processed_img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    #Finding the edges using canny edge detector
    processed_img = cv2.Canny(orig_image, threshold1=200, threshold2=300)
    # Blurring/Smoothening the image to reduce the noise
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    #Adding a mask
    vertices = np.array([[0,450], [320,350], [960,350], [1280,450], [1280,1000], [0,1000]])
    processed_img = region_of_interest(processed_img, [vertices])
    #Finding line edges
    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 10, 10)

    return processed_img

def run(model):
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    pred_counter = 0
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    while True:
        img = np.array(ImageGrab.grab(bbox=(0,40,1280,1000)))       #Window read
        
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
        
        X_test[0] = img
        pred_val = model.predict(X_test)
        lane_pos = cv2.resize(pred_val[0], (640, 480))
        cv2.imshow('lane_detected', lane_pos)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        

if __name__ == '__main__':
    
    
    model = tf.keras.models.load_model('gta_lane_model.h5')
    run(model)

    '''
    test_img = imread("D:/Deep_Learning/AI plays gta/lane dataset/train/lane_img_" + str(img_no) + ".jpg")
    test_img = resize(test_img, (128, 128), mode='constant', preserve_range=True)
    X_test[0] = test_img
    imshow(np.squeeze(X_test[0]))
    pred_val = model.predict(X_test) 
    imshow(np.squeeze(pred_val[0]))
    '''
