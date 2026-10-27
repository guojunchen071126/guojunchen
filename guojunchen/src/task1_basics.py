import cv2
import sys
#读取图片
image1 = cv2.imread("../data/nailong.jpg")
image2 = cv2.imread("../data/mieba.jpg")
imagehui1 = cv2.imread("../data/nailong.jpg",cv2.IMREAD_GRAYSCALE)
imagehui2 = cv2.imread("../data/mieba.jpg",cv2.IMREAD_GRAYSCALE)

if image1 is None:
    sys.exit("Could not read the image.")
if image2 is None:
    sys.exit("Could not read the image.")
if imagehui1 is None:
        sys.exit("Could not read the image.")
if imagehui2 is None:
        sys.exit("Could not read the image.")

result1 =cv2.resize(image1,(300, 300))
result2 =cv2.resize(image2,(300, 300))
#裁剪工具
crop1 = image1[50:300,60:500]
crop2 = image2[30:1000,500:1440]
#打印图像宽、高和通道数
print(image1.shape)
print(image2.shape)
print(imagehui1.shape)
print(imagehui2.shape)
cv2.imshow("result1",result1)
cv2.imshow("crop", crop1)
cv2.imshow("imagehui1",imagehui1)
cv2.imshow("imagehui2",imagehui2)
cv2.imshow("result2", result2)
cv2.imshow("crop", crop2)
cv2.imwrite("../Outputs/task1/result1.caijian.jpg",crop1)
cv2.imwrite("../Outputs/task1/result1.suofang.jpg",result1)
cv2.imwrite("../Outputs/task1/result2.caijian.jpg",crop2)
cv2.imwrite("../Outputs/task1/result2.suofang.jpg",result2)
cv2.imwrite("../Outputs/task1/result1.huidu.jpg",imagehui1)
cv2.imwrite("../Outputs/task1/result2.huidu.jpg",imagehui2)
cv2.waitKey(0)