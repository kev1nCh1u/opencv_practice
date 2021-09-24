import numpy as np
import cv2 as cv
import glob

################ parameter #############################
chessboardSize = (9,6)
frameSize = (640,480)

path = "img/stereo_calibration/new/"
fname = "05.jpg" # 5 6

cameraMatrix1 = np.transpose(np.array([
                      [811.941913143685,	0,	0],
                      [0,	810.875335328820,	0],
                      [327.158512162055,	235.970970735321,	1],
                     ]))
distCoeffs1 = np.array([
                      [0.0527660165864733, 0.601359116399581, 0, 0],
                     ])
cameraMatrix2 = np.transpose(np.array([
                      [811.050002556267,	0,	0],
                      [0,	810.196367616702,	0],
                      [332.300586327942,	256.537534709126,	1],
                     ]))
distCoeffs2 = np.array([
                      [0.0623655915852615, 0.642469145517834, 0, 0],
                     ])
imageSize = (640,480)
stereoR = np.transpose(np.array([
                      [0.998736733831271,	-0.0292308739663654,	0.0408716589240964],
                      [0.0285583073999248,	0.999448512612113,	0.0169438400572838],
                      [-0.0413444019728984,	-0.0157552100778719,	0.999020727403043],
                     ]))
stereoT = np.array([
                      [138.517482167507],
                      [-4.39411590791419],
                      [3.07093845260097],
                     ])

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
                    
# Show the frames
cv.imshow("frame left", frame_left)
cv.imshow("frame right", frame_right) 

# Hit "q" to close the window
cv.waitKey(0)

cv.destroyAllWindows()
