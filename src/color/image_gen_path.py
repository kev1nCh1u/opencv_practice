import cv2  
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("img/red-heart.png")  

 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)  
cv2.imshow("gray", gray)  
cv2.imshow("binary", binary)  
 
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
cv2.drawContours(img,contours,-1,(0,255,255),3)  
print(contours[0][-3])

cv2.imshow("img", img)  
cv2.waitKey(1)  

contoursNp = np.array(contours).reshape(-1,2)
# print(contoursNp)
path = ([1,1]-contoursNp / [640,480]) * 1 + 0
# print(path[-3])

# show all 2d
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('X')
ax.set_ylabel('Z')

sc = ax.scatter(path[:,0], path[:,1], s=20, label='Marker', c=path[:,0], cmap='jet')
ax.legend()
cbar = plt.colorbar(sc)
cbar.set_label('RMSE')
plt.show()