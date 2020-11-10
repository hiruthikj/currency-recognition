#!usr/bin/env python
from utils import *
from matplotlib import pyplot as plt
import cv2
import numpy as np

import glob
import os
from pathlib import Path

MIN_MATCH_COUNT = 10
KERNEL_SIZE = 3

def main():
    print('Currency Recognition Program starting...')

    max_matches = 0
    best_i = -1
    best_kp = 0
    sum_kp = 0

    test_dir = 'testing_images'
    currency_dir = 'currency_images'
    test_image_name = 'test_50_2.jpg'

    training_set = [
        img for img in glob.glob(os.path.join(currency_dir, "*.jpg"))
    ]
    training_set_name = [
        Path(img_path).stem for img_path in training_set
    ]

    test_image_loc = os.path.join(test_dir, test_image_name)
    test_img = read_img(test_image_loc)

    # resizing to display
    image_to_view = resize_img(test_img, 0.4)
    display('INPUT', image_to_view)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    display('After Converting to grayscale', test_img)

    test_img = cv2.equalizeHist(test_img.astype(np.uint8))
    display('After Histogram Equalization', test_img)

    # test_img = cv2.GaussianBlur(test_img, (KERNEL_SIZE,KERNEL_SIZE), sigmaX=0)
    test_img = cv2.bilateralFilter(test_img, KERNEL_SIZE, KERNEL_SIZE*2, (KERNEL_SIZE+1)//2);
    display('After bilateralBlur', test_img)

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(test_img, mask=None)

    for i in range(len(training_set)):
        train_img = cv2.imread(training_set[i],cv2.IMREAD_GRAYSCALE)
        kp2, des2 = orb.detectAndCompute(train_img, mask = None)

        # brute force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Match descriptors
        all_matches = bf.knnMatch(des1, des2, k=2)

        good = []

        # store all the good matches as per Lowe's ratio test.
        for m, n in all_matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        sum_kp += len(kp2)

        num_matches = len(good)
        if num_matches > max_matches:
            max_matches = num_matches
            best_i = i
            best_kp = kp2

        print(i, ' ', training_set[i], ' ', len(good))

    if max_matches >= MIN_MATCH_COUNT:
        print(f'Good Match Found!\n{training_set_name[best_i]} has maximum matches of {max_matches}')

        train_img = cv2.imread(training_set[best_i])
        match_img = cv2.drawMatchesKnn(test_img, kp1, train_img, best_kp, good, None)#, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        note = str(training_set_name[best_i])
        print(f'\nDetected denomination: {note}')

        plt.imshow(match_img), plt.title('DETECTED MATCH'), plt.show()

    else:
        print('No Good Matches, closest one has {max_matches} matches')
        train_img = cv2.imread(training_set[best_i])
        closest_match = cv2.drawMatchesKnn(test_img, kp1, train_img, best_kp, good, 4)

        note = str(training_set_name[best_i])
        print(f'\nPredicted denomination: {note}')


        plt.imshow(closest_match), plt.title('CLOSEST MATCH'), plt.show()

    print('Program exited')

if __name__ == '__main__':
    main()
