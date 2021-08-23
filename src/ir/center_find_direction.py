###################################################################################
# center_find_direction
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

# find center and Direction to array (np(5,1,2))
def findDirection(center_points):
    # closest point with center
    closestDis = float('Inf')
    closestNum = 0
    comparePoint = center_points[0].squeeze()
    disArr = []
    for i in range(1, len(center_points)):
        point = center_points[i].squeeze()
        dis = distance(comparePoint, point)
        if(dis < closestDis):
            closestDis = dis
            closestNum = i
        disArr.append(dis)
    print('\n center_dis\n', disArr, '\n')

    center_points[[1, closestNum]] = center_points[[closestNum, 1]]
    print('\n center_points\n', center_points, '\n')

    print('\n test\n', center_points[[closestNum, 1]], '\n')

    # closest farthest point with center closest point
    closestDis = float('Inf')
    closestNum = 0
    farthestDis = 0
    farthestNum = 0
    comparePoint = center_points[1].squeeze()
    disArr = np.zeros((4, 1, 2), np.int32)
    disArr = []
    for i in range(2, len(center_points)):
        point = center_points[i].squeeze()
        dis = distance(comparePoint, point)
        if(dis < closestDis):
            closestDis = dis
            closestNum = i
        if(dis > farthestDis):
            farthestDis = dis
            farthestNum = i
        disArr.append(dis)
    print('\n dis\n', disArr, '\n')

    center_points[[2, closestNum]] = center_points[[closestNum, 2]]
    center_points[[4, farthestNum]] = center_points[[farthestNum, 4]]
    print('\n center_points\n', center_points, '\n')

    return center_points


# main
if __name__ == '__main__':

    print('\n opencv version:', cv2.__version__)

    center_points = np.array([
        [(349, 289)],
        [(303, 347)],
        [(389, 346)],
        [(439, 345)],
        [(305, 176)],
    ])
    print('\n org_center_points:\n', center_points)

    center_points = findDirection(center_points)
    print('\n find_center_points:\n', center_points)