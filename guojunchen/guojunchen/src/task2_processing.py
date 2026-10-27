
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("../data/fennu.jpg")
imagehui = cv2.imread("../data/fennu.jpg", cv2.IMREAD_GRAYSCALE)
gaosilvbo = cv2.GaussianBlur(image, (101, 101), 0)
ret,thresh = cv2.threshold(imagehui,127,255,0)
_,thresh2 = cv2.threshold(imagehui,127,255,cv2.THRESH_BINARY_INV)
Cannybianyuan = cv2.Canny(imagehui,20,100)
kernel = np.ones((5,5), np.uint8)
erosion = cv2.erode(thresh2,kernel)
dilation = cv2.dilate(thresh2,kernel)
cv2.imshow("gaosilvbo",image)
cv2.imshow("thresh",thresh)
cv2.imshow("gaosilvbo",gaosilvbo)
cv2.imshow("Cannybianyuan",Cannybianyuan)
cv2.imshow("erosion",erosion)
cv2.imshow("dilation",dilation)
cv2.imwrite("../Outputs/task2/imagehui.jpg",imagehui)
cv2.imwrite("../Outputs/task2/erosion.jpg",erosion)
cv2.imwrite("../Outputs/task2/dilation.jpg",dilation)
cv2.imwrite("../Outputs/task2/gaosilvbo.jpg",gaosilvbo)
cv2.imwrite("../Outputs/task2/thresh.jpg",thresh)
cv2.imwrite("../Outputs/task2/gaosilvbo.jpg",Cannybianyuan)
cv2.waitKey(0)
cv2.destroyAllWindows()

