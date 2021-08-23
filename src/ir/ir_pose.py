import cv2
import numpy as np
import yaml
import glob
import center_find_direction
import farthest_find_direction

print('\n opencv version:', cv2.__version__)

########################################################################################
# load img
########################################################################################
if 0:
    cap = cv2.VideoCapture(0)
else:
    path = "img/"
    fname = "ir_led_4.bmp"
    cap = cv2.imread(path + fname)

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
if 0:
    cv2.imshow('gray_image', gray_image)
    cv2.imshow('thresh', thresh)
    cv2.waitKey(0)

###################################################################################
# findContours
###################################################################################
imgResult = cap.copy()
contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgResult, contours, -1, (0, 255, 0), 1)
if 0:
    cv2.imshow('imgResult',imgResult)
    cv2.waitKey(0)

###################################################################################
# find circle center
###################################################################################
# kevin value points
ir_points = np.zeros((4, 1, 2), np.int32)
i = 0

for c in contours:
    # calculate moments for each contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # kevin save point
    point = np.array([cX, cY], np.int32)
    ir_points[i][0] = point
    i += 1

print('\n ir_points: \n', ir_points)

###################################################################################
# find object center
###################################################################################
center_points = np.zeros((5, 1, 2), np.int32)

M = cv2.moments(ir_points)
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

point = np.array([cX, cY], np.int32)
print('\n center:', point)
center_points[0][0] = point
center_points[1:] = ir_points
print('\n center_points: \n', center_points)
print()

###################################################################################
# find direction
###################################################################################
center_points = center_find_direction.findDirection(center_points)
ir_points = farthest_find_direction.findDirection(ir_points)

center_points[1:] = ir_points

###################################################################################
# draw point
###################################################################################
i = 0
for p in center_points:
    cX = p[0][0]
    cY = p[0][1]
    # calculate x,y coordinate of center
    cv2.circle(imgResult, (cX, cY), 2, (0, 0, 255), -1)
    strVar = str(i)
    cv2.putText(imgResult, strVar, (cX + 15, cY + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
    i += 1
if 0:
    cv2.imshow('imgResult', imgResult)
    cv2.waitKey(0)

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

corners2 = center_points[1:].astype(np.float32)
print('\n corners2:')
print('corners2 type:', type(corners2))
print('corners2 shape:', corners2.shape)
print(corners2)

objp = np.array([
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (-2.0, 0.0, 0.0),
    (-2.0, -4.0, 0.0),
])

objp = np.array([
    (0.0, 0.0, 0.0),
    (2.0, 171.0, 0.0),
    (84.0, 170.0, 0.0),
    (134.0, 169.0, 0.0),
])

print('\n objp: \n', objp)

axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, -100]]).reshape(-1, 3)

# Find the rotation and translation vectors.
ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
# project 3D points to image plane
imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
imgResult = draw(imgResult, corners2, imgpts)

# imgpts, jac = cv2.projectPoints(np.array([(0.0, 0.0, -200)]), rvecs, tvecs, mtx, dist)
# cv2.line(imgResult, point1, point2, (255,255,255), 2)

########################################################################################
# end
########################################################################################

cv2.imshow('imgResult', imgResult)
k = cv2.waitKey(0) & 0xFF
if k == ord('s') or True:
    cv2.imwrite(path + 'result/' + fname + '_result' + '.png', imgResult)
cv2.destroyAllWindows()
