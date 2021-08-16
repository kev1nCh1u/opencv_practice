import cv2
import numpy as np
import yaml

print(cv2.__version__)

# cap = cv2.VideoCapture(0)
cap = cv2.imread("img\ir_led_4.bmp")

########################################################################################
# load calibration file 
########################################################################################
with open("calibration_matrix.yaml", "r") as f:
    data = yaml.safe_load(f)

mtx =  data['camera_matrix']
dist = data['dist_coeff']

mtx = np.array(mtx)
dist = np.array(dist)

print('\n camera_matrix: \n', mtx)
print('\n dist_coeff: \n', dist)

###################################################################################
# HSV mask
###################################################################################
# # Take each frame
# # _, frame = cap.read()
# frame = cap.copy()
# # Convert BGR to HSV
# hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
# # define range of blue color in HSV
# lower_blue = np.array([92,134,69])
# upper_blue = np.array([131,255,255])
# # Threshold the HSV image to get only blue colors
# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# # Bitwise-AND mask and original image
# res = cv2.bitwise_and(frame,frame, mask= mask)

# # cv2.imshow('frame',frame)
# # cv2.imshow('mask',mask)
# # cv2.imshow('res',res)

###################################################################################
# HSV mask
###################################################################################
img = cap.copy()

# convert the image to grayscale
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# convert the grayscale image to binary image
ret, thresh = cv2.threshold(gray_image, 80, 255, 0)
cv2.imshow('gray_image', gray_image)
cv2.imshow('thresh', thresh)
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
# find center
###################################################################################
# kevin value
points = np.zeros((4, 1, 2), np.int32)
i = 0

for c in contours:
	# calculate moments for each contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])

	# kevin save point
	point = np.array([cX, cY], np.int32)
	points[i][0] = point
	i += 1

	# calculate x,y coordinate of center
	cv2.circle(imgResult, (cX, cY), 2, (0, 255, 0), -1)
	xy = 'x:' + str(cX) + ' y:' + str(cY)
	cv2.putText(imgResult, xy, (cX + 2, cY + 2),
				cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

cv2.imshow('imgResult',imgResult)

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
