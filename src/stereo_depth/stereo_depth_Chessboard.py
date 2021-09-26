import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt

# Function for stereo vision and depth estimation
# import triangulation as tri
# import calibration

# Mediapipe for face detection
# import mediapipe as mp
import time

# find_depth
def find_depth(right_point, left_point, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        # f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)
        f_pixel = f

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    disparity = abs(x_left-x_right)      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    # zDepth = (baseline*f_pixel)/disparity             #Depth in [mm]
    zDepth = (baseline*f_pixel)/disparity            #Depth in [mm]

    return zDepth

# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# undistortRectify remap
def undistortRectify(frameR, frameL):

    # Undistort and rectify images
    undistortedL= cv2.remap(frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR= cv2.remap(frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return undistortedR, undistortedL

# mediapipe
# mp_facedetector = mp.solutions.face_detection
# mp_draw = mp.solutions.drawing_utils

# Open both cameras
cap_left =  cv2.VideoCapture(4)
cap_right = cv2.VideoCapture(0)   
# cap_left =  cv2.VideoCapture(2, cv2.CAP_DSHOW)
# cap_right = cv2.VideoCapture(4, cv2.CAP_DSHOW)                    


# Stereo vision setup parameters
frame_rate = 120    #Camera frame rate (maximum at 120 fps)
B = 138.596414801303               #Distance between the cameras [mm]
f = 811.060887393561                        #Camera lense's focal length [mm]
alpha = 0        #Camera field of view in the horisontal plane [degrees]


# Main program loop with face detector and depth estimation using stereo vision
# with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# succes_right, frame_right = cap_right.read()
# succes_left, frame_left = cap_left.read()

path = "img/stereo_calibration/new/"
fname = "06.jpg" # 5 6
frame_left = cv2.imread(path + '1/' + fname)
frame_right = cv2.imread(path + '2/' + fname)

# cv2.imshow('frame_left',frame_left)
# cv2.imshow('frame_right',frame_right)
# cv2.waitKey(0)

################## CALIBRATION #########################################################

frame_right, frame_left = undistortRectify(frame_right, frame_left)
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
    # corners_left = cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
    # corners_right = cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)

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
    cv2.putText(frame_right, "Lost !!!", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
    cv2.putText(frame_left, "Lost !!!", (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)

else:
    # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
    # All formulas used to find depth is in video presentaion
    depth = find_depth(center_point_right, center_point_left, frame_right, frame_left, B, f, alpha)

    cv2.putText(frame_right, "Dis: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(frame_left, "Dis: " + str(round(depth,1)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
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

if ret_left  or ret_right:
    cv2.circle(frame_right, center_point_right.astype(np.int32), 10, (0, 0, 255), -1)
    cv2.circle(frame_left, center_point_left.astype(np.int32), 10, (0, 0, 255), -1)

############################### draw green line ##########################################################
imageSize = (np.shape(stereoMapL_x)[1], np.shape(stereoMapL_x)[0])
gap = 27
for i in range(1, int(imageSize[1] / gap) + 1):
    y = gap * i
    cv2.line(frame_left, (0, y), (imageSize[0], y), (0, 255, 0), 1)
    cv2.line(frame_right, (0, y), (imageSize[0], y), (0, 255, 0), 1)

vis = np.concatenate((frame_left, frame_right), axis=1) # mix

# Show the frames
cv2.imshow("frame left", frame_left)
cv2.imshow("frame right", frame_right)
cv2.imshow("vis SubPix" + fname, vis)


# Hit "q" to close the window
cv2.waitKey(0)


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()