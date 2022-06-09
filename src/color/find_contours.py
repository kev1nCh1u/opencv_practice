import cv2  
 
img = cv2.imread("img/contours.jpg")  
img = cv2.imread("img/red-heart.png")  

 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
ret, binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)  
cv2.imshow("gray", gray)  
cv2.imshow("binary", binary)  
 
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)  
cv2.drawContours(img,contours,-1,(0,255,255),3)  
print(contours)

cv2.imshow("img", img)  
cv2.waitKey(0)  