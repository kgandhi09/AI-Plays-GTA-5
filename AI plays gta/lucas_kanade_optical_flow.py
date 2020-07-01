import cv2
import numpy as np
from PIL import ImageGrab

#Mouse function
def select_point(event, x, y, flags, params):
    global point, point_selected, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        point_selected = True
        old_points = np.array([[x,y]], dtype=np.float32)
        
def lucas_kanade():
    WORLD_HEIGHT = 1280
    WORLD_WIDTH = 1024
    
    #Create Old Frame
    frame = np.array(ImageGrab.grab(bbox=(0,40,WORLD_HEIGHT,WORLD_WIDTH)))
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Lucas Kanade Params
    lk_params = dict(winSize = (10,10),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", select_point)
    
    point_selected = False
    point = ()
    old_points = np.array([[]])
    
    while True:
        frame = np.array(ImageGrab.grab(bbox=(0,40,WORLD_HEIGHT,WORLD_WIDTH)))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if point_selected == True:
            cv2.circle(frame, point, 5, (0,0,255), 2)        
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
            old_gray = gray_frame.copy()
            x,y = new_points.ravel()
            cv2.circle(frame, (x, y), 5, (0,255,0), -1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.release()
            cv2.destroyAllWindows()
            break
        
if __name__ == "__main__":
    lucas_kanade()