import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

img = cv2.imread('face.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

fig_img = plt.figure(figsize=(10,10))
col = 6
row = 4

count = 1
for theta in range(1,3):
    theta = theta /4. * np.pi
    for sigma in (3,5):
        for lamda in np.arange(np.pi/4, np.pi, np.pi/4):
            for gamma in(0.05, 0.5):
                kernel = cv2.getGaborKernel((50, 50), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                fig_img.add_subplot(row, col, count)
                plt.imshow(kernel)
                count +=1

plt.show()