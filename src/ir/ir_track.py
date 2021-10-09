import cv2
import numpy as np
# cap = cv2.VideoCapture(0)
cap = cv2.imread("img/ir/Pic_2021_10_09_104654_1.bmp")
cv2.imshow('org cap',cap)
cv2.waitKey(0)
cv2.destroyAllWindows()

###################################################################################
# threshold mask
###################################################################################
img = cap.copy()

# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image
# ret,thresh = cv2.threshold(gray_image,80,255,0)
ret,thresh = cv2.threshold(gray_image,24,255,0)

cv2.imshow('gray_image',gray_image)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
mask = thresh.copy()

###################################################################################
# findContours
###################################################################################
imgResult = cap.copy()
contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgResult,contours,-1,(0,0,255),1)  

cv2.imshow('imgResult',imgResult)
cv2.waitKey(0)

###################################################################################
# find center
###################################################################################
for c in contours:
	# calculate moments for each contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	
    # calculate x,y coordinate of center
	cv2.circle(imgResult, (cX, cY), 2, (0, 0, 255), -1)
	cv2.putText(imgResult, 'x:'+str(cX)+' y:'+str(cY), (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
cv2.imshow('imgResult',imgResult)

cv2.waitKey(0)
cv2.destroyAllWindows()