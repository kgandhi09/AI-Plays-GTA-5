import cv2
import time

#opening the video file

cap = cv2.VideoCapture('D:\Deep_Learning\AI plays gta\lane dataset\Lane video.mp4')
i=0
init_time = time.time()
main_time = time.time()
name_count = 1
while(cap.isOpened()):
    curr_time = time.time()
    ret, frame = cap.read()
    if(curr_time - init_time > 0.15):
        cv2.imwrite('D:/Deep_Learning/AI plays gta/lane dataset/train/lane_img_' + str(name_count) + '.jpg', frame)
        init_time = curr_time
        name_count += 1

    i+=1

cap.release()
cv2.destroyAllWindows()