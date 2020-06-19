import numpy as np
import cv2

vertices_left = []
vertices_right = []
masking_done = False
masked = []
counter = 83                   # counter: index of image in train folder to start with 

def create_mask(event, x, y, flags, params):
 
    global vertices_left, vertices_right, masking_done, masked, counter
    
    black_img_dim = img.shape
    plane_black_img = np.zeros(black_img_dim, np.uint8)
    mask = np.zeros_like(img)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        masking_done = False
        point = (x,y)
        vertices_left.append(point)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        vertices_left = np.array(vertices_left)
        
        if vertices_left.size != 0:
            cv2.fillPoly(mask, [vertices_left], (255, 255, 255))
            masked = cv2.bitwise_and(img, mask)
            print('mask created')
            
        if vertices_left.size == 0:
            masked = cv2.bitwise_and(img, plane_black_img)
            print('empty mask created')
            
        masking_done = True
        vertices_left = []

def run():
    global img, masking_done, counter
    while True:
        img = cv2.imread("D:/Deep_Learning/AI plays gta/lane dataset/train/lane_img_" + str(counter) + ".jpg")
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("image",img)
        cv2.setMouseCallback("image", create_mask)
        if masking_done == True:
            cv2.imwrite("D:/Deep_Learning/AI plays gta/lane dataset/masked/lane_masked_temp_" + str(counter) + ".jpg", masked)
            counter += 1
            masking_done = False
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
if __name__ == "__main__":
    run()
