import cv2
import numpy as np
import yaml
import glob

print(cv2.__version__)

# cap = cv2.VideoCapture(0)
fname = "img\ir_led_4_turn.bmp"
cap = cv2.imread(fname)

########################################################################################
# load calibration file
########################################################################################
with open("calibration_matrix.yaml", "r") as f:
    data = yaml.safe_load(f)

mtx = data['camera_matrix']
dist = data['dist_coeff']

mtx = np.array(mtx)
dist = np.array(dist)

print('\n camera_matrix: \n', mtx)
print('\n dist_coeff: \n', dist)

###################################################################################
# threshold mask
###################################################################################
img = cap.copy()

# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# convert the grayscale image to binary image
ret, thresh = cv2.threshold(gray_image, 80, 255, 0)
# cv2.imshow('gray_image', gray_image)
# cv2.imshow('thresh', thresh)
# cv2.waitKey(0)
mask = thresh.copy()

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
# find circle center
###################################################################################
# kevin value points
points = np.zeros((5, 1, 2), np.int32)
i = 1

for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # kevin save point
    point = np.array([cX, cY], np.int32)
    points[i][0] = point
    i += 1

print('\n points: \n', points)

###################################################################################
# find obj center
###################################################################################
M = cv2.moments(points[1:])
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

point = np.array([cX, cY], np.int32)
print('point:', point)
points[0][0] = point
print('\n points: \n', points)

###################################################################################
# draw point
###################################################################################
i = 0
for p in points:
	cX = p[0][0]
	cY = p[0][1]
    # calculate x,y coordinate of center
	cv2.circle(imgResult, (cX, cY), 2, (0, 0, 255), -1)
	xy = str(i) + ' x:' + str(cX) + ' y:' + str(cY)
	cv2.putText(imgResult, xy, (cX + 15, cY + 2),
				cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
	i += 1
# cv2.imshow('imgResult', imgResult)

###################################################################################
# find point end
###################################################################################
# cv2.waitKey(1000)
cv2.destroyAllWindows()

########################################################################################
# draw xyz
########################################################################################
def draw(img, corners, imgpts):
    corner = tuple(corners[0].astype(np.int32).ravel())
    img = cv2.line(img, corner, tuple(
        imgpts[0].astype(np.int32).ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(
        imgpts[1].astype(np.int32).ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(
        imgpts[2].astype(np.int32).ravel()), (0, 0, 255), 5)
    return img


########################################################################################
# pnp pose
########################################################################################
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

corners2 = points.astype(np.float32)
print('\n corners2:')
print('corners2 type:', type(corners2))
print('corners2 shape:', corners2.shape)
print(corners2)

objp = np.array([
	(0.0, 0.0, 0.0),
    (1.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (-1.0, -1.0, 0.0),
])
print('\n objp: \n', objp)

axis = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, -2]]).reshape(-1, 3)

# Find the rotation and translation vectors.
ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
# project 3D points to image plane
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)


########################################################################################
# end
########################################################################################
imgResult = draw(imgResult, corners2, imgpts)
cv2.imshow('imgResult', imgResult)
k = cv2.waitKey(0) & 0xFF
if k == ord('s'):
	cv2.imwrite(fname + '_result' +'.png', imgResult)
cv2.destroyAllWindows()
