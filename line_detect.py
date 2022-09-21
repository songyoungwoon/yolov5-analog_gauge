import cv2
import numpy as np

img = cv2.imread('./runs/imgs/220601_222330.jpg')
canny = cv2.Canny(img, 250, 255)

cv2.imshow('img', canny)
cv2.waitKey()

lines = cv2.HoughLinesP(canny, 1, np.pi / 180., 30, minLineLength=5, maxLineGap=10)
print(lines)

dst = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

if lines is not None:  # 라인 정보를 받았으면
    for i in range(lines.shape[0]):
        pt1 = (lines[i][0][0], lines[i][0][1])  # 시작점 좌표 x,y
        pt2 = (lines[i][0][2], lines[i][0][3])  # 끝점 좌표, 가운데는 무조건 0
        cv2.line(dst, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('src', img)
        cv2.imshow('dst', dst)
        cv2.waitKey()

