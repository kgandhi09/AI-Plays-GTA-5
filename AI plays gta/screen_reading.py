from PIL import ImageGrab
from win32 import win32gui

window = win32gui.FindWindow(None, 'File Explorer')
win32gui.SetForeroundWindow(window)
dimensions = win32gui.GetWindowRect(window)

