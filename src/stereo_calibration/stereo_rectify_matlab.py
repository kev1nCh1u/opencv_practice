import numpy as np
import cv2 as cv
import glob

################ parameter #############################
chessboardSize = (9,6)
frameSize = (640,480)

path = "img/stereo_calibration/new/"
fname = "05.jpg" # 5 6

cameraMatrix1 = np.transpose(np.array([
                      [811.060887393561,	0,	0],
                      [0,	810.207157388433,	0],
                      [332.297650989743,	256.534222795515,	1],
                     ]))
distCoeffs1 = np.array([
                      [0.0623395705385842,	0.642653150386603, 0, 0],
                     ])
cameraMatrix2 = np.transpose(np.array([
                      [811.953028214802,	0,	0],
                      [0,	810.886363430986,	0],
                      [327.156355343648,	235.966780713027,	1],
                     ]))
distCoeffs2 = np.array([
                      [0.0527316443172304,	0.601816281601748, 0, 0],
                     ])
imageSize = np.array([480	,640])[::-1]
stereoR = np.transpose(np.array([
                      [0.998736795485593,	0.0285581790591670,	-0.0413430012455876],
                      [-0.0292307527930174,	0.999448504197982,	-0.0157559686331825],
                      [0.0408702389804808,	0.0169445526716074,	0.999020773407859],
                     ]))
stereoT = np.transpose(np.array([
                      [-138.596414801303,	0.383887538737156,	2.59021558075892],
                     ]))

########## Stereo Rectification #################################################
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, stereoR, stereoT)

stereoMapL = cv.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, rectL, projMatrixL,imageSize, cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, rectR, projMatrixR,imageSize, cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

stereoMapL_x = stereoMapL[0]
stereoMapL_y = stereoMapL[1]
stereoMapR_x = stereoMapR[0]
stereoMapR_y = stereoMapR[1]

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()

############################### remap ##########################################################
frame_left = cv.imread(path + '1/' + fname)
frame_right = cv.imread(path + '2/' + fname)

frame_left = cv.remap(frame_left, stereoMapL_x, stereoMapL_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)
frame_right = cv.remap(frame_right, stereoMapR_x, stereoMapR_y, cv.INTER_LANCZOS4, cv.BORDER_CONSTANT, 0)

############################### draw green line ##########################################################
gap = 27
for i in range(1, int(imageSize[1] / gap) + 1):
    y = gap * i
    cv.line(frame_left, (0, y), (imageSize[0], y), (0, 255, 0), 1)
    cv.line(frame_right, (0, y), (imageSize[0], y), (0, 255, 0), 1)

vis = np.concatenate((frame_left, frame_right), axis=1) # mix

# Show the frames
cv.imshow("frame left", frame_left)
cv.imshow("frame right", frame_right)
cv.imshow("vis", vis)

# Hit "q" to close the window
cv.waitKey(0)

cv.destroyAllWindows()
