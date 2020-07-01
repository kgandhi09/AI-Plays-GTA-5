from PIL import ImageGrab
import cv2
import numpy as np
from directkeys import PressKey, ReleaseKey, W, A, S, D
import tensorflow as tf
from statistics import mean

def conv_pred_to_world(world, pred, world_height, world_width):
    try:
        
        white_pixels_pred = np.argwhere(pred >= 0.05)
        lane_coords_x = []
        lane_coords_y = []
        # (x,y) of pred world are : (white_pixels_pred[pixel][1],white_pixels_pred[pixel][0])
        for pixel in range(len(white_pixels_pred)):
            x_world = (white_pixels_pred[pixel][1]*world_height)/128
            y_world = (white_pixels_pred[pixel][0]*world_width)/128
            lane_coords_x.append(int(x_world))
            lane_coords_y.append(int(y_world))
            cv2.circle(world, (int(x_world),int(y_world)), 2, (0,0,255), 2)
        
        return lane_coords_x, lane_coords_y
        
    except:
        pass
    
def drive_trajectory(world, lane_coords_x, lane_coords_y):
    point = (int(mean(lane_coords_x) + 1.5*(int(mean(lane_coords_x)))), int(mean(lane_coords_y)))
    return point
    
def run(model):
    WORLD_HEIGHT = 1280
    WORLD_WIDTH = 1024
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    while True:
        
        world = np.array(ImageGrab.grab(bbox=(0,40,WORLD_HEIGHT,WORLD_WIDTH)))       #Window read
        world_resized = cv2.resize(world, (IMG_HEIGHT, IMG_WIDTH))        
        X_test[0] = world_resized
        pred_val = model.predict(X_test)
        
        lane_coords_x, lane_coords_y = conv_pred_to_world(world, pred_val[0], WORLD_HEIGHT, WORLD_WIDTH)
        point = drive_trajectory(world, lane_coords_x, lane_coords_y)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        
if __name__ == '__main__':
    
    model = tf.keras.models.load_model('gta_lane_model.h5')
    run(model)
