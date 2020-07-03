from PIL import ImageGrab
import cv2
import numpy as np
from directkeys import PressKey, ReleaseKey,W, A, D
import tensorflow as tf
from statistics import mean
import time

def select_point_lucas_kanade(event, x, y, flags, params):
    global local_point, point_selected
    try:
        if event == cv2.EVENT_LBUTTONDOWN:
            local_point = (x,y)
            point_selected = True
    except:
        pass

def lucas_kanade_optical_flow(world):
    global local_point_selected, point
    try:
        if point_selected == True:
            cv2.circle(world, local_point, 5, (255,0,0), 5)
        return local_point
    except:
        pass

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
    try:
        point = (int(mean(lane_coords_x)), int(mean(lane_coords_y)))
        cv2.circle(world, point, 5, (0,255,0), 5)
        return point
    except:
        pass
    
def drive(local_x, drive_x):
    global init_time
    try:      
        curr_time = time.time()
        if curr_time - init_time > 1:
            PressKey(W)
            time.sleep(0.5)
            ReleaseKey(W)
            init_time = curr_time
        if local_x > drive_x:
            PressKey(A)
            ReleaseKey(A)         
        elif(local_x < drive_x):
            PressKey(D)
            ReleaseKey(D)
    except:
        pass

def run(model):
    WORLD_HEIGHT = 1280
    WORLD_WIDTH = 1024
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
 
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    flag = False
   
    while True:
        world = np.array(ImageGrab.grab(bbox=(0,40,WORLD_HEIGHT,WORLD_WIDTH)))
        local_xy = lucas_kanade_optical_flow(world)
        
        world_resized = cv2.resize(world, (IMG_HEIGHT, IMG_WIDTH))        
        X_test[0] = world_resized
        pred_val = model.predict(X_test)
        
        lane_coords_x, lane_coords_y = conv_pred_to_world(world, pred_val[0], WORLD_HEIGHT, WORLD_WIDTH)
        drive_xy = drive_trajectory(world, lane_coords_x, lane_coords_y)
        
        if point_selected and not flag:
            print("AI will take over in T-5\n")
            time.sleep(5)
            flag = True
            print("Hello I am AI, leave everything up to me now!")
        
        if point_selected and flag and ( local_xy != None and drive_xy != None):
            drive(local_xy[0], drive_xy[0])
        
        cv2.imshow("AI plays GTA 5", world)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break       
   
if __name__ == '__main__':
    #------------------------------------------------------------------------------------ 
    # Select a local point to localize the player
    local_point = ()
    point_selected = False
    init_time = time.time()
    cv2.namedWindow("AI plays GTA 5")
    cv2.setMouseCallback("AI plays GTA 5", select_point_lucas_kanade)
    print("Localize your player by clicking on it\n")
    
    #-----------------------------------------------------------------------------------
    model = tf.keras.models.load_model('gta_lane_model.h5')
    run(model)