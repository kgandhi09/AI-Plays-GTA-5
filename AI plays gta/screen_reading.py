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
from lane_detection import draw_lanes

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

def run():
    while True:
        window_read = np.array(ImageGrab.grab(bbox=(0,40,1280,1000)))
        screen = process_image(window_read)
        cv2.imshow('window', screen)


        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    run()

