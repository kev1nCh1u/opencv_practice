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
def find_depth(left_point, right_point, baseline, focal):
    x_right = right_point[0]
    x_left = left_point[0]

    # displacement between left and right frames [pixels]
    disparity = abs(x_left - x_right)
    zDepth = (baseline * focal) / disparity            # z depth in [mm]

    return zDepth

###############################################################################################
# calcu_world_point
###############################################################################################
def calcu_world_point(point, z_depth, focal):
    world_points = np.zeros((3), np.float)
    x_cam = point[0]
    y_cam = point[1]
    world_points[0] = (x_cam * z_depth) / focal
    world_points[1] = (y_cam * z_depth) / focal
    world_points[2] = z_depth
    return world_points

###############################################################################################
# undistortRectify remap
###############################################################################################
def undistortRectify(stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, frameL, frameR):

    # Undistort and rectify images
    undistortedL = cv2.remap(
        frameL, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    undistortedR = cv2.remap(
        frameR, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)

    return undistortedL, undistortedR


###################################################################################
# main
###################################################################################
def main():
    while 1:
        capFlag = 0
        save_path = 'img/result/'

        # Camera parameters to undistort and rectify images
        cv_file = cv2.FileStorage()
        cv_file.open('param/stereoMap.xml', cv2.FileStorage_READ)
        stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
        stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
        stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
        stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

        # Stereo vision setup parameters
        fs = cv2.FileStorage("param/matlab_stereo_param.yaml", cv2.FILE_STORAGE_READ)
        baseline = abs(fs.getNode("TranslationOfCamera2").mat().ravel()[0])
        focalLength = fs.getNode("FocalLength").mat().ravel()[0]

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
        frame_left, frame_right = undistortRectify(
            stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y, frame_left, frame_right)
        # cv2.imshow('frame_left',frame_left)
        # cv2.imshow('frame_right',frame_right)
        # cv2.waitKey(0)

        # Convert the BGR image to gray
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, (9, 6), None)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, (9, 6), None)

        # if find point
        if ret_left and ret_right:
            corners_left = cv2.cornerSubPix(gray_left,corners_left,(11,11),(-1,-1),criteria)
            corners_right = cv2.cornerSubPix(gray_right,corners_right,(11,11),(-1,-1),criteria)

            conerNum = 0
            center_point_left = corners_left[conerNum].ravel()
            center_point_right = corners_right[conerNum].ravel()

            center_point_left1 = corners_left[1].ravel()
            center_point_right1 = corners_right[1].ravel()

            print('center_point_left', center_point_left)
            print('center_point_right', center_point_right)

        else:
            center_point_left = None
            center_point_right = None
            print('No Chessboard !!!')

        # if find point, show x y on image
        if ret_left or ret_right:
            cv2.circle(frame_left, center_point_left.astype(
                np.int32), 10, (0, 0, 255), -1)
            cv2.circle(frame_right, center_point_right.astype(
                np.int32), 10, (0, 0, 255), -1)

            cv2.circle(frame_left, center_point_left1.astype(
                np.int32), 10, (255, 0, 0), -1)
            cv2.circle(frame_right, center_point_right1.astype(
                np.int32), 10, (255, 0, 0), -1)

            # find depth
            depth = find_depth(center_point_left, center_point_right, baseline, focalLength)
            depth = round(depth, 1)
            print("Depth:", depth)
            text = "Depth: " + str(round(depth, 1))
            cv2.putText(frame_left, text,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # find depth1
            depth1 = find_depth(center_point_left1.ravel(), center_point_right1.ravel(), baseline, focalLength)
            depth1 = round(depth1, 1)
            print("Depth:", depth1)
            text = "Depth: " + str(round(depth1, 1))
            cv2.putText(frame_left, text,
                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            world_points = np.zeros((2,3), np.float)
            # find world point
            world_points[0] = calcu_world_point(center_point_left, depth, focalLength)
            world_points[0,0] = round(world_points[0,0], 1)
            world_points[0,1] = round(world_points[0,1], 1)
            print('x_world, y_world :' , world_points[0])
            text = "X:" + str(round(world_points[0,0], 1)) + " Y:" + str(round(world_points[0,1], 1))
            cv2.putText(frame_left, text,
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # find world point1
            world_points[1] = calcu_world_point(center_point_left1, depth1, focalLength)
            world_points[1,0] = round(world_points[1,0], 1)
            world_points[1,1] = round(world_points[1,1], 1)
            print('x_world, y_world :' , world_points[1])
            text = "X:" + str(round(world_points[1,0], 1)) + " Y:" + str(round(world_points[1,1], 1))
            cv2.putText(frame_left, text,
                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # find distance
            distance = ((world_points[0,0] - world_points[1,0])**2 + (world_points[0,1] - world_points[1,1])**2 + (world_points[0,2] - world_points[1,2])**2)**0.5
            text = "Distance:" + str(round(distance, 1))
            cv2.putText(frame_left, text,
                        (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # If no point can be caught in one camera show text "TRACKING LOST"
        elif not ret_left or not ret_right:
            cv2.putText(frame_left, "Lost !!!", (75, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

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

        # if s save image
        elif inputKey == ord('s'):
            current_time = time.time() # catch time
            filename = save_path + 'chess_vis_' + str(current_time) + '.jpg'
            cv2.imwrite(filename, vis)
            print('\nSave:' , filename, '\n')

    # Release and destroy all windows before termination
    cap_right.release()
    cap_left.release()

    cv2.destroyAllWindows()

# if main
if __name__ == '__main__':
    main()
