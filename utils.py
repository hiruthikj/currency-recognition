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

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    # print(len(channels))
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def getAutoEdge(img, sigma=0.33):
    v = np.median(img)
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    img_edges = cv2.Canny(img, lower_thresh, upper_thresh)
    return img_edges

