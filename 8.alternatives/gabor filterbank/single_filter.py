import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('face.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ksize = 50
sigma = 50 #size
theta = 1*np.pi/4 #rotation
lamda = 52*np.pi/2#wave length
gamma = 1 #shape

kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)

fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
kernel_resized = cv2.resize(kernel, (400,400))

# plt.imshow(kernel)
# plt.show()


#cv2.imshow('original', img)
#cv2.imshow('filtered', fimg)
cv2.imshow('kernel', kernel_resized)
cv2.waitKey()
cv2.destroyAllWindows()