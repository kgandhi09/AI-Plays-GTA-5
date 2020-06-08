from PIL import ImageGrab
from win32 import win32gui
import cv2
import numpy as np
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked

def process_image(orig_image):
    processed_img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    vertices = np.array([[0,600], [320,330], [960,330], [1280,600], [1280,1000], [680,500], [600,500], [0,1000]])
    processed_img = region_of_interest(processed_img, [vertices])
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

