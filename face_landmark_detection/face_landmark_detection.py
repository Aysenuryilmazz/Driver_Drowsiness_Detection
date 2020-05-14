#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 22:34:18 2020

@author: aysenur
"""


#pip install opencv-python
#pip install cmake dlib

#You can download a trained facial shape predictor from: 
#http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#or train your own by http://dlib.net/train_shape_predictor.py.html
#
#pip install --upgrade imutils

####### TUTORIAL real time face landmark detection:
#https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/
from imutils.video import FileVideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# initialize the video stream with faster method of imutils than opencv
print("[INFO] camera sensor warming up...")
vs = FileVideoStream('yawning.avi').start()
time.sleep(1.0)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
    
    	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw them on the image
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
    
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()