# -*- coding: utf-8 -*-

from directkeys import PressKey, ReleaseKey, W
import time

time.sleep(5)
while True:
    PressKey(W)
    time.sleep(1)