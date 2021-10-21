import sys
import cv2
import numpy as np
import time
import imutils
from matplotlib import pyplot as plt
import time

###############################################################################################
# find_depth
###############################################################################################
def find_depth(right_point, left_point, frame_right, frame_left, baseline, f):

    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = f
    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # displacement between left and right frames [pixels]
    disparity = abs(x_left - x_right)
    zDepth = (baseline * f_pixel) / disparity            # z depth in [mm]

    return zDepth

###############################################################################################
# find_depth
###############################################################################################
def calcu_world_point(x_cam, y_cam, z_depth, focal):

    # x_world = focal * x_cam / z_depth
    # y_world = focal * y_cam / z_depth

    x_world = (x_cam * z_depth) / focal
    y_world = (y_cam * z_depth) / focal

    return x_world, y_world

###############################################################################################
# undistortRectify remap
###############################################################################################
def undistortRectify(stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, frameR, frameL):

    # Undistort and rectify images
    undistortedL = cv2.remap(
        frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR = cv2.remap(
        frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return undistortedR, undistortedL


###################################################################################
# main
###################################################################################
def main():
    while 1:
        capFlag = 0

        # Camera parameters to undistort and rectify images
        cv_file = cv2.FileStorage()
        cv_file.open('param/stereoMap.xml', cv2.FileStorage_READ)

        stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
        stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
        stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
        stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

        # Stereo vision setup parameters
        frame_rate = 120  # Camera frame rate (maximum at 120 fps)
        B = 138.596414801303  # Distance between the cameras [mm]
        f = 811.060887393561  # Camera lense's focal length [mm]

        # termination criteria for cornerSubPix
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Open both cameras
        if capFlag:
            cap_left = cv2.VideoCapture(4)
            cap_right = cv2.VideoCapture(0)
            # cap_left =  cv2.VideoCapture(2, cv2.CAP_DSHOW)
            # cap_right = cv2.VideoCapture(4, cv2.CAP_DSHOW)
            succes_right, frame_right = cap_right.read()
            succes_left, frame_left = cap_left.read()
            if not succes_right or not succes_left:
                break
            else:
                print('Cap read success...')

        # open both picture
        if not capFlag:
            path = "img/stereo_calibration/new/"
            fname = "06.jpg"  # 5 6
            frame_left = cv2.imread(path + '1/' + fname)
            frame_right = cv2.imread(path + '2/' + fname)

        # cv2.imshow('frame_left',frame_left)
        # cv2.imshow('frame_right',frame_right)
        # cv2.waitKey(0)

        # calibration
        frame_right, frame_left = undistortRectify(
            stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, frame_right, frame_left)
        # cv2.imshow('frame_left',frame_left)
        # cv2.imshow('frame_right',frame_right)
        # cv2.waitKey(0)

        # storge start time to calcu fps
        start = time.time()

        # Convert the BGR image to RGB
        # frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2RGB)
        # frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2RGB)

        # Convert the BGR image to gray
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (9, 6), None)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, (9, 6), None)
        # ret_left, corners_left = 0,0
        # ret_right, corners_right = 0,0

        # if find point
        if ret_left and ret_right:
            corners_left = cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
            corners_right = cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)

            conerNum = 1
            center_point_right = corners_right[conerNum].ravel()
            center_point_left = corners_left[conerNum].ravel()

            print('center_point_right', center_point_right)
            print('center_point_left', center_point_left)

        else:
            center_point_right = None
            center_point_left = None

            print('No Chessboard !!!')

        # # If no point can be caught in one camera show text "TRACKING LOST"
        if not ret_left or not ret_right:
            cv2.putText(frame_right, "Lost !!!", (75, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_left, "Lost !!!", (75, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            # Function to calculate depth of object. Outputs vector of all depths in case of several balls.
            # All formulas used to find depth is in video presentaion
            depth = find_depth(center_point_right, center_point_left,
                            frame_right, frame_left, B, f)
            depth = round(depth, 1)
            print("Depth:", depth)
            text = "Dis: " + str(round(depth, 1))
            cv2.putText(frame_right, text,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_left, text,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            x_world, y_world = calcu_world_point(center_point_left[0], center_point_left[1], depth, f)
            x_world = round(x_world, 1)
            y_world = round(y_world, 1)
            print('x_world, y_world :' , x_world, y_world)
            text = "X:" + str(round(x_world, 1)) + " Y:" + str(round(y_world, 1))
            cv2.putText(frame_left, text,
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_right, text,
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # fps calculate
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime

        # show fps
        # print("FPS: ", fps)
        # cv2.putText(frame_right, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        # cv2.putText(frame_left, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
        # cv2.putText(frame_right, f'time:' + '{:.3f}'.format(totalTime) + 's', (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        # cv2.putText(frame_left, f'time:'+ '{:.3f}'.format(totalTime) + 's', (50,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # if find point, show x y on image
        if ret_left or ret_right:
            cv2.circle(frame_right, center_point_right.astype(
                np.int32), 10, (0, 0, 255), -1)
            cv2.circle(frame_left, center_point_left.astype(
                np.int32), 10, (0, 0, 255), -1)

        # draw green line
        imageSize = (np.shape(stereoMapL_x)[1], np.shape(stereoMapL_x)[0])
        gap = 27
        for i in range(1, int(imageSize[1] / gap) + 1):
            y = gap * i
            cv2.line(frame_left, (0, y), (imageSize[0], y), (0, 255, 0), 1)
            cv2.line(frame_right, (0, y), (imageSize[0], y), (0, 255, 0), 1)

        # mix to show on one picture
        vis = np.concatenate((frame_left, frame_right), axis=1)

        # Show the frames
        cv2.imshow("frame left", frame_left)
        cv2.imshow("frame right", frame_right)
        cv2.imshow("vis SubPix" + fname, vis)

        # Hit "q" to close the window
        inputKey = cv2.waitKey(0) & 0xFF

        # 若按下 q 鍵則離開迴圈
        if inputKey == ord('q'):
            break

    # Release and destroy all windows before termination
    cap_right.release()
    cap_left.release()

    cv2.destroyAllWindows()

# if main
if __name__ == '__main__':
    main()
