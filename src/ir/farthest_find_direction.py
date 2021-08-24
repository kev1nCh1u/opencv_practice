###################################################################################
# farthest_find_direction
# by kevin
###################################################################################
import cv2
import numpy as np
import yaml
import glob

# Pythagorean distance
def distance(point1, point2):
    x = point1[0] - point2[0]
    y = point1[1] - point2[1]
    ans = (x ** 2 + y ** 2) ** 0.5
    return ans

# find farthest Direction to array (np(5,1,2))
def findDirection(inputPoints):
    # find farthest point for all
    inputPoints_size = len(inputPoints)
    disArr = np.zeros((inputPoints_size, inputPoints_size), np.int32)
    disArrSum = [0] * inputPoints_size
    farthestNum = 0
    farthestDis = 0
    for i in range(inputPoints_size):
        for j in range(inputPoints_size):
            if(i != j):
                disArr[i][j] = distance(inputPoints[i].squeeze(), inputPoints[j].squeeze())
                disArrSum[i] += disArr[i][j]
        if(disArrSum[i] > farthestDis):
            farthestDis = disArrSum[i]
            farthestNum = i
    
    print('\n disArr:\n', disArr)
    print('\n farthestNum:', farthestNum)

    inputPoints[[0, farthestNum]] = inputPoints[[farthestNum, 0]]
    print('\n inputPoints\n', inputPoints, '\n')

    # sort out other point
    for i in range(inputPoints_size - 1):
        for j in range(1, inputPoints_size - 1):
            # cloeset
            if(disArr[farthestNum][j] > disArr[farthestNum][j + 1]):
                disArr[farthestNum][j], disArr[farthestNum][j + 1] = disArr[farthestNum][j + 1], disArr[farthestNum][j]
                inputPoints[[j, j + 1]] = inputPoints[[j + 1, j]]

    print('\n short inputPoints\n', inputPoints, '\n')

    return inputPoints


# main
if __name__ == '__main__':

    print('\n opencv version:', cv2.__version__)  

    inputPoints = np.array([
        [(303, 347)],
        [(389, 346)],
        [(439, 345)],
        [(305, 176)],
    ])
    print('\n org_inputPoints:\n', inputPoints)

    inputPoints = findDirection(inputPoints)
    print('\n find_inputPoints:\n', inputPoints)