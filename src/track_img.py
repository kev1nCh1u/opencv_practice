import cv2
import numpy as np
# cap = cv2.VideoCapture(0)
cap = cv2.imread("img\smarties.png")

###################################################################################
# HSV mask
###################################################################################
# Take each frame
# _, frame = cap.read()
frame = cap.copy()
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# define range of blue color in HSV
lower_blue = np.array([92, 134, 69])
upper_blue = np.array([131, 255, 255])
# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame, frame, mask=mask)

# cv2.imshow('frame',frame)
# cv2.imshow('mask',mask)
# cv2.imshow('res',res)

###################################################################################
# findContours
###################################################################################
imgResult = cap.copy()
# gray = cv2.cv2tColor(res,cv2.COLOR_BGR2GRAY)
# ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(
    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgResult, contours, -1, (0, 255, 0), 1)

# cv2.imshow('gray',gray)
# cv2.imshow('imgResult',imgResult)

###################################################################################
# find center
###################################################################################
points = np.zeros((3, 1, 2), np.int32)
i = 0
for c in contours:

    # calculate moments for each contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    point = np.array([cX, cY], np.int32)
    points[i][0] = point

    # calculate x,y coordinate of center
    cv2.circle(imgResult, (cX, cY), 2, (0, 255, 0), -1)
    xy = 'x:' + str(cX) + ' y:' + str(cY)
    cv2.putText(imgResult, xy, (cX + 2, cY + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    i += 1

cv2.imshow('imgResult', imgResult)

###################################################################################
# find center again
###################################################################################
M = cv2.moments(points)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# calculate x,y coordinate of center
cv2.circle(imgResult, (cX, cY), 2, (0, 0, 255), -1)
xy = 'x:' + str(cX) + ' y:' + str(cY)
cv2.putText(imgResult, xy, (cX + 2, cY + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

cv2.imshow('imgResult', imgResult)

###################################################################################
# end
###################################################################################
cv2.waitKey(0)
cv2.destroyAllWindows()
