import numpy as np
import cv2
from matplotlib import pyplot as plt

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

capFlag = 0

# Open both cameras
if capFlag:
    cap_right = cv2.VideoCapture(0)                    
    cap_left =  cv2.VideoCapture(4)

    # cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)                    
    # cap_left =  cv2.VideoCapture(4, cv2.CAP_DSHOW)

    if not(cap_right.isOpened() and cap_left.isOpened()):
        exit()
else:
    path = "img/stereo_calibration/ball/"
    fname = "1/01.jpg"
    fname2 = "2/01.jpg"
    cap_right = cv2.imread(path + fname2)
    cap_left = cv2.imread(path + fname)

# cv2.imshow("Left", cap_left)
# cv2.imshow("right", cap_right)
# cv2.waitKey(0)

if capFlag:
    succes_right, frame_right = cap_right.read()
    succes_left, frame_left = cap_left.read()
else:
    frame_right = cap_right
    frame_left = cap_left

# Undistort and rectify images
frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

# cv2.imshow("Left", frame_left)
# cv2.imshow("right", frame_right)
# cv2.waitKey(0)


# left_image = cv2.imread('items_l.png', cv2.IMREAD_GRAYSCALE)
# right_image = cv2.imread('items_r.png', cv2.IMREAD_GRAYSCALE)
left_image = frame_left
right_image = frame_right

# Convert the BGR image to gray
gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Left", gray_left)
# cv2.imshow("right", gray_right)
# cv2.waitKey(0)

stereo = cv2.StereoBM_create(numDisparities=0, blockSize=11)
# For each pixel algorithm will find the best disparity from 0
# Larger block size implies smoother, though less accurate disparity map
depth = stereo.compute(gray_left, gray_right)

print(depth)

# cv2.imshow("Left", gray_left)
# cv2.imshow("right", gray_right)

plt.imshow(depth)
plt.axis('off')
plt.show()
