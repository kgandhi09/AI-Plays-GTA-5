import numpy as np
import cv2

vertices = []
masking_done = False
masked = []
counter = 1

def create_mask(event, x, y, flags, params):
    
    global vertices, masking_done, masked, counter
    if event == cv2.EVENT_LBUTTONDOWN:
        masking_done = False
        point = (x,y)
        vertices.append(point)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        vertices = np.array(vertices)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [vertices], (255, 255, 255))
        masked = cv2.bitwise_and(img, mask)
        masking_done = True
        vertices = []

        
def run():
    global img, masking_done, counter
    while True:
        img = cv2.imread("D:/Deep_Learning/AI plays gta/test_lane_dataset/train/lane_img_" + str(counter) + ".jpg")
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("image",img)
        cv2.setMouseCallback("image", create_mask)
        if masking_done == True:
            cv2.imwrite("D:/Deep_Learning/AI plays gta/test_lane_dataset/masked/lane_masked_" + str(counter) + ".jpg", masked)
            counter += 1
            masking_done = False
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
if __name__ == "__main__":
    run()
