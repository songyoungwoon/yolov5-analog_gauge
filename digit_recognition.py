import time
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
#% matplotlib inline
import os
from PIL import Image

model = tf.keras.models.load_model('digit_models/aaa.h5')
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
def digit_prediction(img):
    # cv2.imshow("original", img)
    # cv2.waitKey()
    original = img.copy()
    # 이미지 흑백처리
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 이미지 블러
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # 이미지 내의 경계 찾기
    ret, img_th = cv2.threshold(img_gray, 20, 255, cv2.THRESH_TOZERO_INV)
    # 디지털 숫자 간극 없애기
    img_th = cv2.GaussianBlur(img_th, (21, 21), 0)

    # 전처리한 이미지 show
    # cv2.imshow("img_th", img_th)
    # cv2.waitKey()

    # Contour(윤곽) 찾기
    contours, hierachy = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 경계를 직사각형으로 찾기
    rects = [cv2.boundingRect(each) for each in contours]
    # 왼쪽부터 읽어와야 하므로 정렬 왼쪽에 있는 경계 순서대로 정렬(rects[0]인 x으로 정렬)
    rects = sorted(rects)
    # 직사각형 영역 추출 확인하기
    rectangle = []
    for rect in rects:
        if rect[2] > 20 and rect[3] > 50:
            rectangle.append(rect)
            cv2.circle(img, (rect[0], rect[1]), 10, (0, 0, 255), -1)
            cv2.circle(img, (rect[0] + rect[2], rect[1] + rect[3]), 10, (0, 0, 255), -1)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

    # 탐지한 이미지 show
    # cv2.imshow("contour_detect", img)
    # cv2.waitKey()

    # 전처리 전 이미지에서 좌표값으로 숫자추출
    # 이전에 처리해놓은 이미지 사용
    img_for_class = original.copy()
    # 최종 이미지 파일용 배열
    digit_imgs = []
    margin_pixel = 0
    # 숫자 영역 추출 및 (28,28,1) reshape
    for rect in rectangle:
        # 숫자영역 추출
        im = img_for_class[rect[1] - margin_pixel:rect[1] + rect[3] + margin_pixel,
             rect[0] - margin_pixel:rect[0] + rect[2] + margin_pixel]
        row, col = im.shape[:2]

        # 정방형 비율을 맞춰주기 위해 변수 이용
        bordersize = max(row, col)
        diff = min(row, col)
        # 이미지의 intensity의 평균을 구함
        bottom = im[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        # border추가해 정방형 비율로 보정
        border = cv2.copyMakeBorder(
            im,
            top=0,
            bottom=0,
            left=int((bordersize - diff) / 2),
            right=int((bordersize - diff) / 2),
            borderType=cv2.BORDER_CONSTANT,
            value=[mean, mean, mean]
        )
        square = border
        # square 사이즈 (28,28)로 축소
        try:
            resized_img = cv2.resize(square, dsize=(120, 120), interpolation=cv2.INTER_AREA)
            digit_imgs.append(resized_img)
        except:
            break

    number = ""
    for i in range(len(digit_imgs)):
        img = digit_imgs[i]
        # cv2.imshow("img", img)
        # cv2.waitKey()

        # rect 이미지 저장 : 학습데이터 추출시 사용
        # u = "runs/rect_imgs/" + str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))+".jpg"
        # time.sleep(1)
        # cv2.imwrite(u, img)

        # 이미지를 784개 흑백 픽셀로 사이즈 변환
        #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # cv2.imshow("img", img)
        # cv2.waitKey()
        input_data = img.reshape(1, 120, 120, 3)

        # 데이터를 모델에 적용할 수 있도록 가공
        # input_data = ((np.array(img) / 255) - 1) * -1

        # 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
        predictions = model.predict(input_data)
        score = tf.nn.softmax(predictions[0])
        number += str(class_names[np.argmax(score)])
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
    return(number)

# 학습용 이미지 추출시 사용
# for j in range(10):
#     for i in range(17, 21, 1):
#         img = cv2.imread(f"rect/{i}.jpg")
#         digit_prediction(img)

# test 용
# img = cv2.imread("rect/4.jpg")
# digit_prediction(img)



