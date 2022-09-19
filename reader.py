import cv2
import numpy as np

def make_middle_line(arr):
    h, w = arr.shape
    mh, mw = h//2, w//2
    for i in range(mh//2):
        arr[mh][mw] = 255
        mh -= 1

def im_rotate(img, degree):
    h, w = img.shape
    crossLine = int(((w * h + h * w) ** 0.5))
    centerRotatePT = int(w / 2), int(h / 2)
    new_h, new_w = h, w
    rotatefigure = cv2.getRotationMatrix2D(centerRotatePT, 360-degree, 1)
    result = cv2.warpAffine(img, rotatefigure, (new_w, new_h))

    return result

def detect_digit_num(zero_angle, max_angle, detect_angle, max_num):
    detect_angle = 360 - detect_angle
    entire_range = abs(max_angle - zero_angle)
    detect_range = abs(detect_angle - zero_angle)
    detect_percent = detect_range/entire_range
    detect_digit_num = max_num*detect_percent
    return round(detect_digit_num, 1)

def digit_num_predict(img, cls):
    image_size = 200

    img = cv2.resize(img, (image_size, image_size))
    canny = cv2.Canny(img, 100, 255)
    virtual_line = np.zeros((image_size,image_size), dtype=np.uint8)
    make_middle_line(virtual_line)

    angle = []
    for i in range(360):
        arr2 = im_rotate(virtual_line, i)
        bit_and = cv2.bitwise_and(canny, arr2)
        angle.append(np.sum(bit_and))

    print("detect_angle :", angle.index(max(angle)))
    print("detect_num :", detect_digit_num(0, 360, angle.index(max(angle)), 12))

    return angle.index(max(angle)), detect_digit_num(0, 360, angle.index(max(angle)), 12)

def angle_predict(img):
    image_size = 200

    img = cv2.resize(img, (image_size, image_size))
    canny = cv2.Canny(img, 100, 255)
    virtual_line = np.zeros((image_size, image_size), dtype=np.uint8)
    make_middle_line(virtual_line)

    cv2.imshow("canny", canny)
    cv2.imshow("line", virtual_line)
    cv2.waitKey()
    angle = []
    for i in range(360):
        for j in range(10):
            arr2 = im_rotate(virtual_line, i+j/10)
            bit_and = cv2.bitwise_and(canny, arr2)
            angle.append(np.sum(bit_and))
    return angle.index(max(angle))/10

img = cv2.imread('./clock/KakaoTalk_20220918_143024275.jpg')
img = cv2.imread('./runs/imgs/220601_222330.jpg')
print(angle_predict(img))
