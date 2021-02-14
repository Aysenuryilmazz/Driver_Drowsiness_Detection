#This class has the aproximate copy of the filterbank in the
#-İlhan U, Ozan A, Erdem U, Levent Ö
#Detection Of Driver Sleepiness And Warning The Driver
#In Real-time Using Image Processing And Machine Learning Techniques (2017)
#Article

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

img = cv2.imread('face.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig_img = plt.figure(figsize=(10,10))
col = 8
row = 5

count = 1
lamda = 2*np.pi/4
for sigma in range(3,18,3):
    lamda += 3*np.pi/4
    for theta in np.arange(np.pi/2, 12*np.pi/8, np.pi/8):
        kernel = cv2.getGaborKernel((31, 31), sigma, theta, lamda, 1, 0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        fig_img.add_subplot(row, col, count)
        plt.imshow(kernel)
        count +=1

plt.show()