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
        lane_coords_xy = []
        # (x,y) of pred world are : (white_pixels_pred[pixel][1],white_pixels_pred[pixel][0])
        for pixel in range(len(white_pixels_pred)):
            x_world = (white_pixels_pred[pixel][1]*world_height)/128
            y_world = (white_pixels_pred[pixel][0]*world_width)/128
            lane_coords_xy.append((int(x_world), int(y_world)))
            cv2.circle(world, (int(x_world),int(y_world)), 2, (0,0,255), 2)
        return lane_coords_xy
    except:
        pass
    
def drive_trajectory(world, lane_coords_xy):
    counter = 0
    sorted_coord_list = []
    sorted_coords_xy = sorted(lane_coords_xy, key = lambda x: x[1])
    for coord in range(len(sorted_coords_xy)):
        if coord != len(sorted_coords_xy)-1:
            if sorted_coords_xy[coord + 1][1] != sorted_coords_xy[coord][1]:
                row_coord_list = sorted_coords_xy[counter:coord+1]
                counter = coord+1
                sorted_coord_list.append(row_coord_list)
        else:
            row_coord_list = sorted_coords_xy[counter:]
            sorted_coord_list.append(row_coord_list)
    trajectory = []
    for row_list in sorted_coord_list:
        sum = 0
        for i in range(len(row_list)):
            sum += row_list[i][0]
        mean_x = sum/len(row_list)
        point = (int(mean_x), row_list[0][1])
        trajectory.append(point)
    for point in range(len(trajectory)):
        cv2.circle(world, trajectory[point], 4, (0,0,0), 4)
        if point != 0:
            cv2.line(world, trajectory[point-1], trajectory[point], (0,255,0), 2)
    return trajectory
    
'''
def kalman_filter(data):
    upd_x = [0]
    state_estimate = [0]
    P = [0]
    upd_P = [0]
    Q = R = 0.1
    K = [0]
    t = [0]

    k = 0
    while k < 201:
        if k > 0: 
            upd_x.append(state_estimate[k-1])
            upd_P.append(P[k-1] + Q)
            K.append(upd_P[k]/(upd_P[k] + R))
            state_estimate.append(upd_x[k] + (K[k] * (data[k] - upd_x[k])))
            P.append( (1 - K[k]) * upd_P[k])
            t.append(0.025*k)   
        k = k + 1
    return t
'''
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
            time.sleep(0.1)
            ReleaseKey(A)         
        elif(local_x < drive_x):
            PressKey(D)
            time.sleep(0.1)
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
        
        lane_coords_xy = conv_pred_to_world(world, pred_val[0], WORLD_HEIGHT, WORLD_WIDTH)
        trajectory = drive_trajectory(world, lane_coords_xy)
        #filtered_trajectory = kalman_filter(trajectory)
        '''
        if point_selected and not flag:
            print("AI will take over in T-5\n")
            time.sleep(5)
            flag = True
            print("Hello I am AI, leave everything up to me now!")
        
            drive(local_xy[0], drive_xy[0])
            pass
        '''
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
    model = tf.keras.models.load_model('gta_lane_model_v2.h5')
    run(model)