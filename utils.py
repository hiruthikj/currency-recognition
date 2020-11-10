# contains utility functions

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

def display(window_name, image):
	screen_res = 1440, 900	# MacBook Air

	scale_width = screen_res[0] / image.shape[1]
	scale_height = screen_res[1] / image.shape[0]
	scale = min(scale_width, scale_height)
	window_width = int(image.shape[1] * scale)
	window_height = int(image.shape[0] * scale)

	# reescale the resolution of the window
	cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(window_name, window_width, window_height)

	# display image
	cv2.imshow(window_name, image)

	# wait for any key to quit the program
	cv2.waitKey()
	cv2.destroyAllWindows()

def resize_img(image, scale):
	res = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_AREA)
	return res
