#!usr/bin/env python
from utils import *
from matplotlib import pyplot as plt
import cv2
import numpy as np

import glob
import os
from pathlib import Path

MIN_MATCH_COUNT = 5
KERNEL_SIZE = 7

test_dir = 'testing_images'
currency_dir = 'currency_images'
test_image_name = 'test_20_2.jpg'
# test_image_name = 'test_100_1.jpg'
# test_image_name = 'Cat03.jpg'

def main():
    print('Currency Recognition Program starting...\n')
    print('ACTUAL denomination', test_image_name)

    training_set = [
        img for img in glob.glob(os.path.join(currency_dir, "*.jpg"))
    ]
    training_set_name = [
        Path(img_path).stem for img_path in training_set
    ]

    test_image_loc = os.path.join(test_dir, test_image_name)
    test_img = cv2.imread(test_image_loc)

    # preprocess image
    test_img = preprocess(test_img)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(test_img, mask=None)

    max_matches = -1
    sum_kp = 0

    for i in range(len(training_set)):
        train_img = cv2.imread(training_set[i])#,cv2.IMREAD_GRAYSCALE)
        train_img = preprocess(train_img, showImages=False)
        kp2, des2 = orb.detectAndCompute(train_img, mask = None)

        # brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Match descriptors
        all_matches = bf.knnMatch(des1, des2, k=2)

        good = []

        # store all the good matches as per Lowe's ratio test.
        for m, n in all_matches:
            if m.distance < 0.7 * n.distance:
                good.append([m])

        sum_kp += len(kp2)

        num_matches = len(good)
        if num_matches > max_matches:
            max_matches = num_matches
            best_i = i
            best_kp = kp2
            best_img = train_img

        print(f'{i+1} \t {training_set[i]} \t {len(good)}')

    kp_perc = len(best_kp)/sum_kp*100
    # if max_matches >= MIN_MATCH_COUNT:
    if kp_perc > 10:
        print(f'\nMatch Found!\n{training_set_name[best_i]} has maximum matches of {max_matches} ({kp_perc}%)')

        match_img = cv2.drawMatchesKnn(test_img, kp1, best_img, best_kp, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        note = training_set_name[best_i]
        print(f'\nDetected denomination: {note}')

        plt.imshow(match_img), plt.title(f'DETECTED MATCH: {note}'), plt.show()

    else:
        print(f'\nNo Good Matches, closest one has {max_matches} matches ({kp_perc}%)')

        closest_match = cv2.drawMatchesKnn(test_img, kp1, best_img, best_kp, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        note = training_set_name[best_i]
        # print(f'\nPredicted denomination: {note}')

        plt.imshow(closest_match), plt.title('NO MATCH'), plt.show()

    print('\nProgram exited')







def preprocess(img, showImages = True):
    #showing the image input    # img = cv2.equalizeHist(img.astype(np.uint8))
    # showImages and display('After Histogram Equalization', img)ed
    img = resize_img(img, 0.4)
    showImages and display('INPUT AFTER RESIZE', img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    showImages and display('After Converting to Grayscale', img)

    # img = cv2.equalizeHist(img.astype(np.uint8))
    # showImages and display('After Histogram Equalization', img)

    # test_img = cv2.GaussianBlur(img, (KERNEL_SIZE,KERNEL_SIZE), sigmaX=0)
    img = cv2.bilateralFilter(img, KERNEL_SIZE, KERNEL_SIZE*2, KERNEL_SIZE//2)
    showImages and display('After Bilateral Blur', img)

    ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    showImages and display('OTSU threshol', img)

    # img = getAutoEdge(img)
    # showImages and display('After Canny Edge Detection', img)
    #
    # contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    # max_area = 0
    # max_i = -1
    #
    # for i in range(len(contours)):
    #     if cv2.contourArea(contours[i]) > max_area:
    #         max_area = cv2.contourArea(contours[i])
    #         max_i = i
    #
    # print(max_i, max_area)
    #
    # showImages and display('Contours', img)
    #
    # cont = cv2.drawContours(img, contours, max_i, (0, 255, 0), 10)
    # showImages and display('Contours 222', cont)


    # kernel = np.ones((5,5),np.uint8)
    # erosion = cv2.erode(img,kernel,iterations = 1)
    # showImages and display('erosion', img)

    return img


def getAutoEdge(img, sigma=0.33):
    v = np.median(img)
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
    img_edges = cv2.Canny(img, lower_thresh, upper_thresh)
    return img_edges


if __name__ == '__main__':
    main()
