########################################################################################
# camera_calibration
# by KevinChiu
########################################################################################

import numpy as np
import cv2 as cv2
import glob
import yaml

print('\n opencv version:', cv2.__version__)

########################################################################################
# findChessboardCorners
########################################################################################
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('img/calibration/*.jpg')
print('\n input images:', len(images))
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        # corners2 = corners
        # print('\n corners \n', corners2)

        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
cv2.destroyAllWindows()
print('\n find images:', len(imgpoints))

########################################################################################
# calibration
########################################################################################
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
print('\n mtx: \n', mtx)
print('\n dist: \n', dist)
print()

########################################################################################
# calibration mean_error
########################################################################################
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print("total error: {}\n".format(mean_error/len(objpoints)))

########################################################################################
# transform the matrix and distortion coefficients to writable lists
########################################################################################
data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()}

########################################################################################
# and save it to a file
########################################################################################
print('save to param/calibration_matrix.yaml\n')
with open("param/calibration_matrix.yaml", "w") as f:
    yaml.dump(data, f)
