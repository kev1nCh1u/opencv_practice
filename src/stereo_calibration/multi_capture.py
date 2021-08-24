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

########################################################################################
# Capture img
########################################################################################
print('videoCapture....')
# cap = cv2.VideoCapture(0)
# cap2 = cv2.VideoCapture(1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
print('finish...\n')

########################################################################################
# read img
########################################################################################
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
        cv2.imwrite('img/stereo_calibration/1/frame1_' + str(int(current_time)) + '.jpg', frame)
        cv2.imwrite('img/stereo_calibration/2/frame2_' + str(int(current_time)) + '.jpg', frame2)
        print('save: ' + str(int(current_time)))

    # 若按下 q 鍵則離開迴圈
    if inputKey == ord('q'):
        break

# 釋放攝影機
cap.release()
ca2.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()
