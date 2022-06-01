import cv2
import numpy as np

def warp(img):
    image_size = 80
    img = cv2.resize(img, (image_size, image_size))
    cv2.imshow("img", img)
    cv2.waitKey()
    canny = cv2.Canny(img, 0, 255)
    h, w = canny.shape
    cv2.imshow("canny", canny)
    # up-down
    break_true = False
    for i in range(h):
        for j in range(w):
            if canny[i][j] == 255:
                up_x, up_y = i, j
                break_true = True
                break
        if break_true:
            break_true = False
            break
    # down-up
    for i in range(h-1, -1, -1):
        for j in range(w):
            if canny[i][j] == 255:
                down_x, down_y = i, j
                break_true = True
                break
        if break_true:
            break_true = False
            break
    # left-right
    for i in range(w):
        for j in range(h):
            if canny[j][i] == 255:
                left_x, left_y = j, i
                break_true = True
                break
        if break_true:
            break_true = False
            break
    # right-left
    for i in range(w-1, -1, -1):
        for j in range(h):
            if canny[j][i] == 255:
                right_x, right_y = j, i
                break_true = True
                break
        if break_true:
            break
    print("up : ", up_x, up_y)
    print("down : ", down_x, down_y)
    print("left : ", left_x, left_y)
    print("right : ", right_x, right_y)
    rect = np.array([[up_x, up_y], [down_x, down_y], [left_x, left_y], [right_x, right_y]], np.float32)
    dst = np.array([[0, image_size//2], [image_size-1, image_size//2], [image_size//2, 0], [image_size//2, image_size-1]], np.float32)
    cv2.waitKey()
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (image_size, image_size))
    cv2.imshow("warped", warped)
    cv2.waitKey()

    #cv2.imwrite("runs/imgs/test.jpg", img)
    return img

img = cv2.imread("test2.jpg")
warp(img)