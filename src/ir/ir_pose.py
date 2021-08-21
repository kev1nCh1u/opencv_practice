import cv2
import numpy as np
import yaml
import glob

print('\n opencv version:', cv2.__version__)

########################################################################################
# load img
########################################################################################
if 0:
    cap = cv2.VideoCapture(0)
else:
    fname = "img\ir_led_4.bmp"
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
# Pythagorean distance
def distance(point1, point2):
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    ans = (x ** 2 + y ** 2) ** 0.5
    return ans


closestDis = float('Inf')
closestNum = 0
comparePoint = center_points[0].squeeze()
disArr = []
for i in range(1, len(center_points)):
    point = center_points[i].squeeze()
    dis = distance(comparePoint, point)
    if(dis < closestDis):
        closestDis = dis
        closestNum = i
    disArr.append(dis)
print('\n center_dis\n', disArr, '\n')

center_points[[1, closestNum]] = center_points[[closestNum, 1]]
print('\n center_points\n', center_points, '\n')

closestDis = float('Inf')
closestNum = 0
farthestDis = 0
farthestNum = 0
comparePoint = center_points[1].squeeze()
disArr = np.zeros((4, 1, 2), np.int32)
disArr = []
for i in range(2, len(center_points)):
    point = center_points[i].squeeze()
    dis = distance(comparePoint, point)
    if(dis < closestDis):
        closestDis = dis
        closestNum = i
    if(dis > farthestDis):
        farthestDis = dis
        farthestNum = i
    disArr.append(dis)
print('\n dis\n', disArr, '\n')

center_points[[2, closestNum]] = center_points[[closestNum, 2]]
center_points[[4, farthestNum]] = center_points[[farthestNum, 4]]
print('\n center_points\n', center_points, '\n')

###################################################################################
# draw point
###################################################################################
i = 0
for p in center_points:
    cX = p[0][0]
    cY = p[0][1]
    # calculate x,y coordinate of center
    cv2.circle(imgResult, (cX, cY), 2, (0, 0, 255), -1)
    xy = str(i) + ' x:' + str(cX) + ' y:' + str(cY)
    cv2.putText(imgResult, xy, (cX + 15, cY + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
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
    (-1.5, 0.0, 0.0),
    (2, 0.0, 0.0),
    (2, -3.5, 0.0),
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
    cv2.imwrite(fname + '_result' + '.png', imgResult)
cv2.destroyAllWindows()
