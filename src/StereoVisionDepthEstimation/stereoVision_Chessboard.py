import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
import triangulation as tri
import calibration

# Mediapipe for face detection
# import mediapipe as mp
import time

# mp_facedetector = mp.solutions.face_detection
# mp_draw = mp.solutions.drawing_utils

# Open both cameras
cap_left =  cv2.VideoCapture(4)
cap_right = cv2.VideoCapture(0)   
# cap_left =  cv2.VideoCapture(2, cv2.CAP_DSHOW)
# cap_right = cv2.VideoCapture(4, cv2.CAP_DSHOW)                    


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 13.8               #Distance between the cameras [cm]
f = 8.2              #Camera lense's focal length [mm]
alpha = 56.6        #Camera field of view in the horisontal plane [degrees]




# Main program loop with face detector and depth estimation using stereo vision
# with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# succes_right, frame_right = cap_right.read()
# succes_left, frame_left = cap_left.read()

frame_left = cv2.imread("img/stereo_calibration/test/1/left05.jpg")
frame_right = cv2.imread("img/stereo_calibration/test/2/right05.jpg")

# cv2.imshow('frame_left',frame_left)
# cv2.imshow('frame_right',frame_right)
# cv2.waitKey(0)

################## CALIBRATION #########################################################

frame_right, frame_left = calibration.undistortRectify(frame_right, frame_left)
# cv2.imshow('frame_left',frame_left)
# cv2.imshow('frame_right',frame_right)
# cv2.waitKey(0)

########################################################################################

# If cannot catch any frame, break
# if not succes_right or not succes_left:                    
#     break

# else:

start = time.time()

# Convert the BGR image to RGB
# frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
# frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

# Convert the BGR image to gray
gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

# Process the image and find faces
# results_right = face_detection.process(frame_right)
# results_left = face_detection.process(frame_left)

# Find the chess board corners
ret_left, corners_left = cv2.findChessboardCorners(gray_left, (9, 6), None)
ret_right, corners_right = cv2.findChessboardCorners(gray_right, (9, 6), None)
# ret_left, corners_left = 0,0
# ret_right, corners_right = 0,0

if ret_left  and ret_right:
    center_point_right = corners_right[0].ravel()
    center_point_left = corners_left[0].ravel()

    print('center_point_right', center_point_right)
    print('center_point_left', center_point_left)

else:
    center_point_right = None
    center_point_left = None

    print('No Chessboard !!!')

# # If no ball can be caught in one camera show text "TRACKING LOST"
if not ret_left  or not ret_right:
    cv2.putText(frame_right, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.putText(frame_left, "TRACKING LOST", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

else:
    # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
    # All formulas used to find depth is in video presentaion
    depth = tri.find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

    cv2.putText(frame_right, "Distance: " + str(round(depth,1)) + 'cm', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame_left, "Distance: " + str(round(depth,1)) + 'cm', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    # Multiply computer value with 205.8 to get real-life depth in [cm]. The factor was found manually.
    print("Depth: ", str(round(depth,1)))



end = time.time()
totalTime = end - start

fps = 1 / totalTime
#print("FPS: ", fps)

# cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
# cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
# cv2.putText(frame_right, f'time:' + '{:.3f}'.format(totalTime) + 's', (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
# cv2.putText(frame_left, f'time:'+ '{:.3f}'.format(totalTime) + 's', (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)                                       

if center_point_right.all() and center_point_left.all() :
    cv2.circle(frame_right, center_point_right.astype(np.int32), 5, (0, 0, 255), -1)
    cv2.circle(frame_left, center_point_left.astype(np.int32), 5, (0, 0, 255), -1)

# Show the frames
cv2.imshow("frame right", frame_right) 
cv2.imshow("frame left", frame_left)


# Hit "q" to close the window
cv2.waitKey(0)


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()