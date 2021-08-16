import numpy as np
import cv2 as cv2
import glob
import yaml

print(cv2.__version__)

########################################################################################
# and save it to a file
########################################################################################
with open("calibration_matrix.yaml", "r") as f:
    data = yaml.safe_load(f)

print('\n camera_matrix: \n', data['camera_matrix'])
print('\n dist_coeff: \n', data['dist_coeff'])

exit()
########################################################################################
# findChessboardCorners
########################################################################################
# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('img/calibration/*.jpg')
print('input images:',len(images))
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # cv2.waitKey(0)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners2, ret)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
cv2.destroyAllWindows()
print('find images:',len(imgpoints))

########################################################################################
# calibration
########################################################################################
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
print('calibrateCamera:')

########################################################################################
# calibration mean_error
########################################################################################
# mean_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     mean_error += error
# print( "total error: {}".format(mean_error/len(objpoints)) )

########################################################################################
# draw xyz
########################################################################################
def draw(img, corners, imgpts):
    corner = tuple(corners[0].astype(np.int32).ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].astype(np.int32).ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].astype(np.int32).ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].astype(np.int32).ravel()), (0,0,255), 5)
    return img

########################################################################################
# pnp pose
########################################################################################
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# objp = np.zeros((6*9,3), np.float32)
# objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)

for fname in glob.glob('img/calibration/WIN_20210810_09_40_14_Pro.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        print('\n objp: \n', objp)
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
        print('corners2 type:',type(corners2))
        print('corners2 shape:',corners2.shape)
        print('corners2 type:',type(corners2[0]))
        print('corners2:',corners2[0].astype(np.int32).ravel())
        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(fname[:6]+'.png', img)
cv2.destroyAllWindows()