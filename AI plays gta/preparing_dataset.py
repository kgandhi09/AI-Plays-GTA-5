import cv2
import time

#opening the video file
cap = cv2.VideoCapture('test_video.mp4')
i=0
init_time = time.time()
main_time = time.time()
name_count = 1
while(cap.isOpened()):
    curr_time = time.time()
    ret, frame = cap.read()
    if(curr_time - init_time > 0.9):
        cv2.imwrite('test' + str(name_count) + '.jpg', frame)
        init_time = curr_time
        name_count += 1
    # if curr_time - main_time > 10:
    #     break
    i+=1

cap.release()
cv2.destroyAllWindows()