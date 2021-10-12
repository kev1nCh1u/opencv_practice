import numpy as np
import cv2


# Camera parameters to undistort and rectify images
cv_file = cv2.FileStorage()
cv_file.open('param/stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

imageSize = (np.shape(stereoMapL_x)[1], np.shape(stereoMapL_x)[0])
capFlag = 0

# Open both cameras
if capFlag:
    cap_right = cv2.VideoCapture(4)                    
    cap_left =  cv2.VideoCapture(2)

    # cap_right = cv2.VideoCapture(0, cv2.CAP_DSHOW)                    
    # cap_left =  cv2.VideoCapture(4, cv2.CAP_DSHOW)

    if not(cap_right.isOpened() and cap_left.isOpened()):
        exit()
else:
    path = "img/stereo_calibration/new/"
    fname = "1/01.jpg"
    fname2 = "2/01.jpg"
    cap_right = cv2.imread(path + fname2)
    cap_left = cv2.imread(path + fname)

while(1):

    if capFlag:
        succes_right, frame_right = cap_right.read()
        succes_left, frame_left = cap_left.read()
    else:
        frame_right = cap_right
        frame_left = cap_left

    # Undistort and rectify images
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    # draw green line
    gap = 27
    for i in range(1, int(imageSize[1] / gap) + 1):
        y = gap * i
        cv2.line(frame_left, (0, y), (imageSize[0], y), (0, 255, 0), 1)
        cv2.line(frame_right, (0, y), (imageSize[0], y), (0, 255, 0), 1)

    vis = np.concatenate((frame_left, frame_right), axis=1) # mix

    # Show the frames
    cv2.imshow("frame left", frame_left)
    cv2.imshow("frame right", frame_right)
    cv2.imshow("vis", vis)


    # Hit "q" to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release and destroy all windows before termination
cap_right.release()
cap_left.release()

cv2.destroyAllWindows()
