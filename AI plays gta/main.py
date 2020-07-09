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
        white_pixels_pred = np.argwhere(pred >= 0.75)
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
    try:
        counter = 0
        sorted_coord_list = []
        sorted_coords_xy = sorted(lane_coords_xy, key = lambda x: x[1])
        for coord in range(len(sorted_coords_xy)):
            if coord != len(sorted_coords_xy)-1:
                if sorted_coords_xy[coord+1][1] != sorted_coords_xy[coord][1]:
                    row_coord_list = sorted_coords_xy[counter:coord+1]
                    counter = coord+1
                    sorted_coord_list.append(row_coord_list)
            else:
                row_coord_list = sorted_coords_xy[counter:]
                sorted_coord_list.append(row_coord_list)
        trajectory = []
        for row_list in sorted_coord_list:
            for coord in range(len(row_list)):
                if coord != 0:
                    if row_list[coord][0] - row_list[coord-1][0] > 200:
                        mean_x = (row_list[coord][0] + row_list[coord-1][0])/2
                        point = (int(mean_x), row_list[0][1])
                        trajectory.append(point)
        temp = []
        for coord in range(len(trajectory)):
            if coord != 0:
                if abs(trajectory[coord][0] - trajectory[coord-1][0]) > 75:
                    temp.append(trajectory[coord-1])
        for coord in temp:
            if coord in trajectory:
                trajectory.remove(coord)
        for point in range(len(trajectory)):
            cv2.circle(world, trajectory[point], 4, (0,0,0), 4)
            if point != 0:
                cv2.line(world, trajectory[point-1], trajectory[point], (0,255,0), 2)
        return True, trajectory 
    except:
        pass

def drive(world, local_xy, trajectory):
    local_x = local_xy[0]
    local_y = local_xy[1]
    tolerance = 30
    drive_flag = True
    try:   
        print(drive_flag)
        if local_y > trajectory[0][1] + tolerance:
            drive_flag = True
            time.sleep(0.4)
        if local_y <= trajectory[0][1] + tolerance:
            print('next iter')
            drive_flag = False
        return drive_flag
    except:
        pass

def run(model):
    WORLD_HEIGHT = 1280
    WORLD_WIDTH = 1024
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    IMG_CHANNELS = 3
 
    X_test = np.zeros((1, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    init_flag = True
    drive_flag = False
    
    while True:  
        if not drive_flag:
            world = np.array(ImageGrab.grab(bbox=(0,40,WORLD_HEIGHT,WORLD_WIDTH)))
            local_xy = lucas_kanade_optical_flow(world)
            
            world_resized = cv2.resize(world, (IMG_HEIGHT, IMG_WIDTH))        
            X_test[0] = world_resized
            pred_val = model.predict(X_test)        
            lane_coords_xy = conv_pred_to_world(world, pred_val[0], WORLD_HEIGHT, WORLD_WIDTH)
            if point_selected and init_flag:
                print("AI will take over in T-5\n")
                time.sleep(5)
                init_flag = False
                print("Hello I am AI, leave everything up to me now!")
            if point_selected:    
                traj_generated, trajectory = drive_trajectory(world, lane_coords_xy)
                drive_flag = True
        if drive_flag:
            reached = drive(world, local_xy, trajectory)
            drive_flag = reached
        
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