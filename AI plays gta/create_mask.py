import numpy as np
import cv2

list_vertices = [[]]
counter = 106
masking_done = False
masked = []
vertices_counter = 0              

def create_mask(event, x, y, flags, params): 
    global counter, masking_done, masked, list_vertices, vertices_counter
    mask = np.zeros_like(img)
    plane_black_image = np.zeros(img.shape, np.uint8)

    if event == cv2.EVENT_LBUTTONDOWN:
        point = (x,y)
        list_vertices[vertices_counter].append(point)
    
    if event == cv2.EVENT_RBUTTONDOWN:
        list_vertices.append([])
        vertices_counter += 1
    
    if event == cv2.EVENT_MBUTTONDOWN:
        
        if list_vertices != [[]]: 
            for vertices in range(len(list_vertices)):
                if len(list_vertices[vertices]) != 0: 
                    list_vertices[vertices] = np.array(list_vertices[vertices])
                    cv2.fillPoly(mask, [list_vertices[vertices]], (255,255,255))
            masked = cv2.bitwise_and(img, mask)
        else:
            masked = cv2.bitwise_and(img, plane_black_image)
        
        masking_done = True
        vertices_counter = 0
        list_vertices = [[]]
        
def run():
    global img, masking_done, counter
    while True:
        img = cv2.imread("D:/Deep_Learning/AI plays gta/lane dataset/train/lane_img_" + str(counter) + ".jpg")
        cv2.namedWindow("image", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("image",img)
        cv2.setMouseCallback("image", create_mask)
        if masking_done == True:
            cv2.imwrite("D:/Deep_Learning/AI plays gta/lane dataset/masked/lane_masked_" + str(counter) + ".jpg", masked)
            print(str(round((counter/3197)*100, 2) ) + "% Data created")
            counter += 1
            masking_done = False
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
if __name__ == "__main__":
    run()
