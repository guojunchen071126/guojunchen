import cv2
import numpy as np
import sys

# 读取 Shrek 图片
image_shilaike = cv2.imread("../data/shilaike.jpeg")
if image_shilaike is None:
    sys.exit("Could not read the image: shilaike")
hsv_shilaike = cv2.cvtColor(image_shilaike, cv2.COLOR_BGR2HSV)

# 读取 Avatar 图片
image_avatar = cv2.imread("../data/avatar.jpg")
if image_avatar is None:
    sys.exit("Could not read the image: avatar")
hsv_avatar = cv2.cvtColor(image_avatar, cv2.COLOR_BGR2HSV)

# 读取程六香图片
image_chengliuxiang = cv2.imread("../data/chengliuxiang.webp")
if image_chengliuxiang is None:
    sys.exit("Could not read the image: chengliuxiang")
hsv_chengliuxiang = cv2.cvtColor(image_chengliuxiang, cv2.COLOR_BGR2HSV)

def empty(a):
    pass

# Shrek
cv2.namedWindow("TrackBars_shilaike")
cv2.resizeWindow("TrackBars_shilaike", 640, 240)
cv2.createTrackbar("H Min", "TrackBars_shilaike", 0, 179, empty)
cv2.createTrackbar("H Max", "TrackBars_shilaike", 179, 179, empty)
cv2.createTrackbar("S Min", "TrackBars_shilaike", 0, 255, empty)
cv2.createTrackbar("S Max", "TrackBars_shilaike", 255, 255, empty)
cv2.createTrackbar("V Min", "TrackBars_shilaike", 0, 255, empty)
cv2.createTrackbar("V Max", "TrackBars_shilaike", 255, 255, empty)

# Avatar
cv2.namedWindow("TrackBars_avatar")
cv2.resizeWindow("TrackBars_avatar", 640, 240)
cv2.createTrackbar("H Min", "TrackBars_avatar", 0, 179, empty)
cv2.createTrackbar("H Max", "TrackBars_avatar", 179, 179, empty)
cv2.createTrackbar("S Min", "TrackBars_avatar", 0, 255, empty)
cv2.createTrackbar("S Max", "TrackBars_avatar", 255, 255, empty)
cv2.createTrackbar("V Min", "TrackBars_avatar", 0, 255, empty)
cv2.createTrackbar("V Max", "TrackBars_avatar", 255, 255, empty)

# 橙留香
cv2.namedWindow("TrackBars_chengliuxiang")
cv2.resizeWindow("TrackBars_chengliuxiang", 640, 240)
cv2.createTrackbar("H Min", "TrackBars_chengliuxiang", 0, 179, empty)
cv2.createTrackbar("H Max", "TrackBars_chengliuxiang", 179, 179, empty)
cv2.createTrackbar("S Min", "TrackBars_chengliuxiang", 0, 255, empty)
cv2.createTrackbar("S Max", "TrackBars_chengliuxiang", 255, 255, empty)
cv2.createTrackbar("V Min", "TrackBars_chengliuxiang", 0, 255, empty)
cv2.createTrackbar("V Max", "TrackBars_chengliuxiang", 255, 255, empty)

kernel = np.ones((5, 5), np.uint8)
AREA_THRESHOLD = 500
records_shilaike = []
records_avatar = []
records_chengliuxiang = []

while True:
    # 处理 Shrek 图片
    h_min_shilaike = cv2.getTrackbarPos("H Min", "TrackBars_shilaike")
    h_max_shilaike = cv2.getTrackbarPos("H Max", "TrackBars_shilaike")
    s_min_shilaike = cv2.getTrackbarPos("S Min", "TrackBars_shilaike")
    s_max_shilaike = cv2.getTrackbarPos("S Max", "TrackBars_shilaike")
    v_min_shilaike = cv2.getTrackbarPos("V Min", "TrackBars_shilaike")
    v_max_shilaike = cv2.getTrackbarPos("V Max", "TrackBars_shilaike")

    lower_shilaike = np.array([h_min_shilaike, s_min_shilaike, v_min_shilaike])
    upper_shilaike = np.array([h_max_shilaike, s_max_shilaike, v_max_shilaike])

    mask_shilaike = cv2.inRange(hsv_shilaike, lower_shilaike, upper_shilaike)
    mask_shilaike = cv2.erode(mask_shilaike, kernel)
    mask_shilaike = cv2.dilate(mask_shilaike, kernel)

    contours_shilaike, hierarchy_shilaike = cv2.findContours(
        mask_shilaike, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_shilaike = image_shilaike.copy()
    records_shilaike = []

    for cnt in contours_shilaike:
        area = cv2.contourArea(cnt)
        if area > AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(result_shilaike, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(result_shilaike, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result_shilaike, f"({cx},{cy}) Area:{int(area)}",
                        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            records_shilaike.append([cx, cy, int(area)])
    cv2.imshow("mask_shilaike", mask_shilaike)
    cv2.imshow("result_shilaike", result_shilaike)
    cv2.imwrite("../Outputs/task3/mask_shilaike.png", mask_shilaike)
    cv2.imwrite("../Outputs/task3/result_shilaike.png", result_shilaike)

    # 处理 Avatar 图片
    h_min_avatar = cv2.getTrackbarPos("H Min", "TrackBars_avatar")
    h_max_avatar = cv2.getTrackbarPos("H Max", "TrackBars_avatar")
    s_min_avatar = cv2.getTrackbarPos("S Min", "TrackBars_avatar")
    s_max_avatar = cv2.getTrackbarPos("S Max", "TrackBars_avatar")
    v_min_avatar = cv2.getTrackbarPos("V Min", "TrackBars_avatar")
    v_max_avatar = cv2.getTrackbarPos("V Max", "TrackBars_avatar")

    lower_avatar = np.array([h_min_avatar, s_min_avatar, v_min_avatar])
    upper_avatar = np.array([h_max_avatar, s_max_avatar, v_max_avatar])

    mask_avatar = cv2.inRange(hsv_avatar, lower_avatar, upper_avatar)
    mask_avatar = cv2.erode(mask_avatar, kernel)
    mask_avatar = cv2.dilate(mask_avatar, kernel)

    contours_avatar, hierarchy_avatar = cv2.findContours(
        mask_avatar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_avatar = image_avatar.copy()
    records_avatar = []
    for cnt in contours_avatar:
        area = cv2.contourArea(cnt)
        if area > AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(result_avatar, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(result_avatar, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result_avatar, f"({cx},{cy}) Area:{int(area)}",
                        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            records_avatar.append([cx, cy, int(area)])
    cv2.imshow("mask_avatar", mask_avatar)
    cv2.imshow("result_avatar", result_avatar)
    cv2.imwrite("../Outputs/task3/mask_avatar.png", mask_avatar)
    cv2.imwrite("../Outputs/task3/result_avatar.png", result_avatar)

    # 处理程六香图片
    h_min_chengliuxiang = cv2.getTrackbarPos("H Min", "TrackBars_chengliuxiang")
    h_max_chengliuxiang = cv2.getTrackbarPos("H Max", "TrackBars_chengliuxiang")
    s_min_chengliuxiang = cv2.getTrackbarPos("S Min", "TrackBars_chengliuxiang")
    s_max_chengliuxiang = cv2.getTrackbarPos("S Max", "TrackBars_chengliuxiang")
    v_min_chengliuxiang = cv2.getTrackbarPos("V Min", "TrackBars_chengliuxiang")
    v_max_chengliuxiang = cv2.getTrackbarPos("V Max", "TrackBars_chengliuxiang")

    lower_chengliuxiang = np.array([h_min_chengliuxiang, s_min_chengliuxiang, v_min_chengliuxiang])
    upper_chengliuxiang = np.array([h_max_chengliuxiang, s_max_chengliuxiang, v_max_chengliuxiang])

    mask_chengliuxiang = cv2.inRange(hsv_chengliuxiang, lower_chengliuxiang, upper_chengliuxiang)
    mask_chengliuxiang = cv2.erode(mask_chengliuxiang, kernel)
    mask_chengliuxiang = cv2.dilate(mask_chengliuxiang, kernel)

    contours_chengliuxiang, hierarchy_chengliuxiang = cv2.findContours(
        mask_chengliuxiang, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_chengliuxiang = image_chengliuxiang.copy()
    records_chengliuxiang = []
    for cnt in contours_chengliuxiang:
        area = cv2.contourArea(cnt)
        if area > AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            cv2.rectangle(result_chengliuxiang, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(result_chengliuxiang, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(result_chengliuxiang, f"({cx},{cy}) Area:{int(area)}",
                        (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            records_chengliuxiang.append([cx, cy, int(area)])
    cv2.imshow("mask_chengliuxiang", mask_chengliuxiang)
    cv2.imshow("result_chengliuxiang", result_chengliuxiang)
    cv2.imwrite("../Outputs/task3/mask_chengliuxiang.png", mask_chengliuxiang)
    cv2.imwrite("../Outputs/task3/result_chengliuxiang.png", result_chengliuxiang)

    if cv2.waitKey(1) & 0xFF == 27:  # 按 Esc 退出
        break

for cx, cy, area in records_shilaike:
    print(f"shilaike: 中心坐标: ({cx}, {cy}), 轮廓面积: {area}")

for cx, cy, area in records_avatar:
    print(f"avatar: 中心坐标: ({cx}, {cy}), 轮廓面积: {area}")

for cx, cy, area in records_chengliuxiang:
    print(f"chengliuxiang: 中心坐标: ({cx}, {cy}), 轮廓面积: {area}")

cv2.destroyAllWindows()
