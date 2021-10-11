import cv2
import numpy as np


###################################################################################
# define
###################################################################################
capFlag = 0

###################################################################################
# ir_track
###################################################################################
def ir_track(frame):
	cv2.imshow('original frame',frame)
	# cv2.waitKey(0)

	###################################################################################
	# threshold mask
	###################################################################################
	img = frame.copy()

	# convert the image to grayscale
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# convert the grayscale image to binary image
	ret,thresh = cv2.threshold(gray_image,80,255,0)
	# ret,thresh = cv2.threshold(gray_image,24,255,0)

	cv2.imshow('gray_image',gray_image)
	cv2.imshow('thresh',thresh)
	# cv2.waitKey(0)
	mask = thresh.copy()

	###################################################################################
	# findContours
	###################################################################################
	imgResult = frame.copy()
	contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(imgResult,contours,-1,(0,0,255),1)  

	cv2.imshow('imgResult',imgResult)
	# cv2.waitKey(0)

	###################################################################################
	# find center
	###################################################################################
	missCount = 0
	for c in contours:
		# calculate moments for each contour
		M = cv2.moments(c)
		if M["m00"] > 0:
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])

			# calculate x,y coordinate of center
			cv2.circle(imgResult, (cX, cY), 2, (0, 0, 255), -1)
			cv2.putText(imgResult, 'x:'+str(cX)+' y:'+str(cY), (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
		else:
			missCount = missCount + 1

	if missCount == len(contours):
		print('miss point')
		cv2.putText(imgResult, 'miss point' , (10,10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
		cv2.imshow('imgResult xy',imgResult)
		cv2.waitKey(0)
	
	else:
		cv2.imshow('imgResult xy',imgResult)
		if capFlag:
			cv2.waitKey(1)
		if not capFlag:
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			exit()

###################################################################################
# main
###################################################################################
def main():

	if capFlag:
		# cap = cv2.VideoCapture(0)
		cap = cv2.VideoCapture('/home/kevin/MVviewer/videos/A5031CU815_4H05A85PAK641B0/Video_2021_10_08_165027_10.avi')

	while 1:
		if capFlag:
			ret, frame = cap.read()

		if not capFlag:
			frame = cv2.imread("img/ir/Pic_2021_10_09_104654_1.bmp")
			ret = True

		if ret == True:
			# print('cap get frame')
			ir_track(frame)
		else:
			print('error no cap frame')
			cv2.destroyAllWindows()
			exit()
		

if __name__ == '__main__':
    main()