import ir_track
import cv2
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt

###################################################################################
# define
###################################################################################
capFlag = 1
imagePath = "img/ir/Pic_2021_10_09_104654_1.bmp"
videoPath = '/home/kevin/MVviewer/videos/A5031CU815_4H05A85PAK641B0/Video_2021_10_08_164744_1.avi'

###################################################################################
# point path
###################################################################################


class PointPath:

    def __init__(self, fileName=0):
        self.fileName = fileName
        self.minPoint = 120 # math.inf
        self.maxPoint = 527 # 0
        self.points = np.zeros((10000, 1, 2), np.int32)
        self.num = 0
        self.saveFlag = False
        self.minFlag = False
        self.maxFlag = False
        self.findMinMaxFlag = True # False
        self.saveDataPath = "data/point_path_data_" + str(self.fileName) + ".csv"
        self.savePlotPath = 'img/result/plot/' + 'point_path_plot_' + str(self.fileName) + '.png'

    def findMinMax(self, point):
        if point[0] < self.minPoint:
            self.minPoint = point[0]
            self.minFlag = self.maxFlag = False
        elif point[0] > self.maxPoint:
            self.maxPoint = point[0]
            self.minFlag = self.maxFlag = False
        elif point[0] == self.minPoint:
            self.minFlag = True
        elif point[0] == self.maxPoint:
            self.maxFlag = True
        elif self.minFlag and self.maxFlag:
            return True

    def savePoint(self, point):
        if not self.findMinMaxFlag:
            self.findMinMaxFlag = self.findMinMax(point)
        
        self.points[self.num][0] = point

        if self.findMinMaxFlag and not self.saveFlag:
            if point[0] == self.minPoint:
                self.saveFlag = True
                print('start save...')
        
        if self.saveFlag == True:
            self.points[self.num][0] = point
            self.num = self.num + 1
            if  point[0] == self.maxPoint:
                # print(self.points[:self.num])
                print('\n\nNum of point', self.num)
                print('MinMax', self.minPoint, self.maxPoint)

                pointsReshape = np.reshape(self.points[:self.num], (-1,2))
                # np.savetxt(self.saveDataPath, pointsReshape, delimiter=",")
                pd.DataFrame(pointsReshape).to_csv(self.saveDataPath)
                print('Save data file to:', self.saveDataPath)

                plt.title('Points')
                plt.xlabel('x axis')
                plt.ylabel('y axis')
                plt.plot(pointsReshape[:,0],pointsReshape[:,1], 'o', markersize=1)
                plt.xlim([0, 640])
                plt.ylim([480, 0])
                plt.savefig(self.savePlotPath)
                print('Save plot image to:', self.savePlotPath)
                plt.show()

                exit()

        return self.minPoint, self.maxPoint

###################################################################################
# main
###################################################################################
def main():
    if capFlag:
        # cap = cv2.VideoCapture(0)
        cap = cv2.VideoCapture(videoPath)

    # fileNum = "{0:0=2d}".format(int(videoPath[-5]))
    fileNum = videoPath[-5]
    PointPath1 = PointPath(fileNum)
    print('file:', fileNum)

    while 1:
        if capFlag:
            ret, frame = cap.read()

        if not capFlag:
            frame = cv2.imread(imagePath)
            ret = True

        if ret == True:
            # print('cap get frame')
            points = ir_track.ir_track(frame, capFlag)
            # print(points[0][0])
            print('Now',points[0][0], 'MinMax',PointPath1.savePoint(points[0][0]), end='\r')
        else:
            print('error no cap frame')
            cv2.destroyAllWindows()
            exit()
		

if __name__ == '__main__':
    main()
