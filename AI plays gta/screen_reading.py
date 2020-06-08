from PIL import ImageGrab
from win32 import win32gui
import cv2
import numpy as np
import time
from directkeys import PressKey, ReleaseKey, W, A, S, D


def process_image(orig_image):
    processed_img = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)
    return processed_img

def run():
    while True:
        window = win32gui.FindWindow(None, 'Grand Theft Auto V')
        win32gui.SetForegroundWindow(window)
        dimensions = win32gui.GetWindowRect(window)

        window_read = np.array(ImageGrab.grab(bbox=dimensions))
        screen = process_image(window_read)
        cv2.imshow('window', screen)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    for i in range(0,4):
        print('waiting for game to start')
        time.sleep(1)

    print('Pressing W')
    PressKey(W)
    time.sleep(5)
    print('Releasing W')
    ReleaseKey(W)
