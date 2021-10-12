import numpy as np
import cv2 as cv2
import glob
import yaml

print(cv2.__version__)

########################################################################################
# load calibration file 
########################################################################################
with open("param/calibration_matrix.yaml", "r") as f:
    data = yaml.safe_load(f)

mtx =  data['camera_matrix']
dist = data['dist_coeff']

mtx = np.array(mtx)
dist = np.array(dist)

print('\n camera_matrix: \n', mtx)
print('\n dist_coeff: \n', dist)

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
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
axis = np.float32([[2,0,0], [0,2,0], [0,0,-2]]).reshape(-1,3)

for fname in glob.glob('img/calibration/WIN_20210810_09_40_14_Pro.jpg'):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        # print('\n objp: \n', objp)
        ret,rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        print('\n corners2:')
        print('corners2 type:',type(corners2))
        print('corners2 shape:',corners2.shape)
        print(corners2)
        print()

        print('\n objp: \n', objp)
        print()

        img = draw(img,corners2,imgpts)
        cv2.imshow('img',img)
        k = cv2.waitKey(0) & 0xFF
        if k == ord('s'):
            cv2.imwrite(fname[:6]+'.png', img)
cv2.destroyAllWindows()