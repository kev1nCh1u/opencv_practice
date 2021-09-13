########################################################################################
# multi_capture
# by kevin
########################################################################################
import cv2
import numpy as np
import yaml
import glob
import time

print('\n opencv version:', cv2.__version__)

save_path = 'img/stereo_calibration/test/'

########################################################################################
# Capture img
########################################################################################
print('videoCapture....')
cap = cv2.VideoCapture(4)
cap2 = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
# cap2 = cv2.VideoCapture(4, cv2.CAP_DSHOW)
print('finish...\n')

if not (cap.isOpened()):
    print("Could not open video device")
    exit()

if not (cap2.isOpened()):
    print("Could not open video device 2")
    exit()

########################################################################################
# read img
########################################################################################
i = 1
while(True):
    # catch time
    current_time = time.time()

    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    ret2, frame2 = cap2.read()

    # 顯示圖片
    cv2.imshow('frame', frame)
    cv2.imshow('frame2', frame2)
    
    inputKey = cv2.waitKey(1) & 0xFF

    # if s save image
    if inputKey == ord('s'):
        # cv2.imwrite(save_path + '1/left' + str(int(current_time)) + '.jpg', frame)
        # cv2.imwrite(save_path + '2/right' + str(int(current_time)) + '.jpg', frame2)
        cv2.imwrite(save_path + '1/left' + "{0:0=2d}".format(i)+ '.jpg', frame)
        cv2.imwrite(save_path + '2/right' + "{0:0=2d}".format(i)+ '.jpg', frame2)
        print('save: ' + str(int(current_time)), "{0:0=2d}".format(i))
        i += 1

    # 若按下 q 鍵則離開迴圈
    if inputKey == ord('q'):
        break

# 釋放攝影機
cap.release()
ca2.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
