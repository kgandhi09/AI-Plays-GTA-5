import numpy as np
import cv2


vertices = []



def create_mask(event, x, y, flags, params):
    
    global vertices
    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        vertices.append(point)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        vertices = np.array(vertices)
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [vertices], 255)
        masked = cv2.bitwise_and(img, mask)
        cv2.imwrite("D:/Deep_Learning/AI plays gta/lane dataset/train_masked/lane_masked_1.jpg", masked)
        vertices = []

        
def run():
    global img
    while True:
        img = cv2.imread("D:/Deep_Learning/AI plays gta/lane dataset/train/lane_img_1.jpg")
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("image",img)
        cv2.setMouseCallback("image", create_mask)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
if __name__ == "__main__":
    run()
