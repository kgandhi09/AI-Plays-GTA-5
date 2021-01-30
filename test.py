# -*- coding: utf-8 -*-

from directkeys import PressKey, ReleaseKey, W, A, D
import time

time.sleep(5)
print('Starting auto control for straight\nw')
init_time = time.time()
right_counter = 0
left_counter = 0
right_flag = True
left_flag = False
init_time_right = time.time()
init_time_left = time.time()

while True:
    curr_time = time.time()
    curr_time_right = time.time()
    curr_time_left = time.time()
    if curr_time - init_time > 0.5:
        PressKey(W)
        time.sleep(0.5)
        ReleaseKey(W)
        init_time = curr_time
    if right_flag:
        PressKey(D)
        time.sleep(0.5)
        ReleaseKey(D)
        time.sleep(0.5)
        if curr_time_right - init_time_right > 1:
            init_time_right = curr_time_right
            init_time_left = curr_time_right
            right_flag = False
            left_flag = True
    if left_flag:
        PressKey(A) 
        time.sleep(0.5)
        ReleaseKey(A)
        time.sleep(0.5)
        if curr_time_left - init_time_left > 0.5:
            init_time_left = curr_time_left
            init_time_right = curr_time_left
            right_flag = True
            left_flag = False
