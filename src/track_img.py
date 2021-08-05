import cv2 as cv
import numpy as np
# cap = cv.VideoCapture(0)
cap = cv.imread("img\smarties.png")

###################################################################################
# HSV mask
###################################################################################
# Take each frame
# _, frame = cap.read()
frame = cap.copy()
# Convert BGR to HSV
hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([92,134,69])
upper_blue = np.array([131,255,255])
# Threshold the HSV image to get only blue colors
mask = cv.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv.bitwise_and(frame,frame, mask= mask)

cv.imshow('frame',frame)
cv.imshow('mask',mask)
cv.imshow('res',res)

###################################################################################
# findContours
###################################################################################
img = cap.copy()
# gray = cv.cvtColor(res,cv.COLOR_BGR2GRAY)  
# ret, binary = cv.threshold(gray,127,255,cv.THRESH_BINARY)  
contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)  
cv.drawContours(img,contours,-1,(0,0,255),1)  

# cv.imshow('gray',gray)
cv.imshow('img',img)

###################################################################################
# findContours
###################################################################################



cv.waitKey(0)
cv.destroyAllWindows()