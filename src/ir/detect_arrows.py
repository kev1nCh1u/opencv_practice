#########################################################################################
# detect_arrows
# https://stackoverflow.com/questions/66718462/python-cv-detect-different-types-of-arrows
#########################################################################################

import cv2
import numpy as np

def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 50, 50)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=1)
    return img_erode

def find_tip(points, convex_hull):
    length = len(points)
    indices = np.setdiff1d(range(length), convex_hull)

    print('\n length\n', length)
    print('\n indices\n', indices)

    for i in range(2):
        j = indices[i] + 2
        if j > length - 1:
            j = length - j
        if np.all(points[j] == points[indices[i - 1] - 2]):
            return tuple(points[j])

img = cv2.imread("img/arrows.png")

contours, hierarchy = cv2.findContours(preprocess(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
    hull = cv2.convexHull(approx, returnPoints=False)
    sides = len(hull)

    print('\n cnt:\n', cnt)
    print('\n peri:\n', peri)
    print('\n approx:\n', approx)
    print('\n hull:\n', hull)
    print('\n hull_squeeze:\n', hull.squeeze())
    print('\n sides:\n', sides)

    # img_approx = img.copy()
    # cv2.drawContours(img_approx, approx, -1, (0, 255, 0), 3)
    # cv2.imshow("img_approx", img_approx)
    # cv2.waitKey(0)

    if 6 > sides > 3 and sides + 2 == len(approx):
        arrow_tip = find_tip(approx[:,0,:], hull.squeeze())
        if arrow_tip:
            cv2.drawContours(img, [cnt], -1, (0, 255, 0), 3)
            cv2.circle(img, arrow_tip, 3, (0, 0, 255), cv2.FILLED)

            print('\n arrow_tip:\n', arrow_tip)
            # cv2.drawContours(img, [approx], -1, (255, 0, 0), 1)
            for point in approx:
                # print('\n point:\n', point.squeeze())
                cv2.circle(img, point.squeeze(), 1, (255, 0, 0), cv2.FILLED)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)

cv2.imshow("Image", img)
cv2.waitKey(0)