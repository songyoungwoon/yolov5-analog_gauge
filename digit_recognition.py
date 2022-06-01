# model.save("/content/mnist_model.h5")

#MNIST 모델
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import cv2
import matplotlib.pyplot as plt
#% matplotlib inline

# 라이브러리 호출
import os
from PIL import Image

# # 변환할 이미지 목록 불러오기
# image_path = './' # 여기만 맞춰서 적어주면 될 것 같음!
# img_list = os.listdir(image_path)  # 디렉토리 내 모든 파일 불러오기
# img_list_jpg = [img for img in img_list if img.endswith(".jpg")]  # 지정된 확장자만 필터링
#
# # 목록의 이미지 리스트업
# img_list_np = []
# for i in img_list_jpg:
#     img = Image.open(image_path + i)
#     img_array = np.array(img)
#     img_list_np.append(img_array)
#     print(i, " 추가 완료 - 구조:", img_array.shape)  # 불러온 이미지의 차원 확인 (세로X가로X색)
#
# # 리스트를 numpy로 변환
# img_np = np.array(img_list_np)
#
# for img in img_np:


def digit_prediction(img):
    # 이미지 읽어오기
    plt.figure(figsize=(15, 12))

    # 이미지 흑백처리
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 블러
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 이미지 내의 경계 찾기
    ret, img_th = cv2.threshold(img_blur, 80, 255, cv2.THRESH_TOZERO_INV)

    # 디지털 숫자 간극 없애기
    img_th = cv2.GaussianBlur(img_th, (11, 11), 0)

    # Contour(윤곽) 찾기
    contours, hierachy = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # 경계를 직사각형으로 찾기
    rects = [cv2.boundingRect(each) for each in contours]

    # 왼쪽부터 읽어와야 하므로 정렬 왼쪽에 있는 경계 순서대로 정렬(rects[0]인 x으로 정렬)
    rects = sorted(rects)

    # 직사각형 영역 추출 확인하기
    rectangle = []
    for rect in rects:
        if rect[2] > 10 and rect[3] > 140:
            rectangle.append(rect)
            cv2.circle(img_blur, (rect[0], rect[1]), 10, (0, 0, 255), -1)
            cv2.circle(img_blur, (rect[0] + rect[2], rect[1] + rect[3]), 10, (0, 0, 255), -1)
            cv2.rectangle(img_blur, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)

    # 이전에 처리해놓은 이미지 사용
    img_for_class = img_gray.copy()

    # 최종 이미지 파일용 배열
    mnist_imgs = []
    margin_pixel = 15

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
            resized_img = cv2.resize(square, dsize=(28, 28), interpolation=cv2.INTER_AREA)
            mnist_imgs.append(resized_img)
        except:
            break

    model = tf.keras.models.load_model('mnist_models/mnist_model.h5')
    #model.summary()

    number = ""
    for i in range(len(mnist_imgs)):
        img = mnist_imgs[i]
        cv2.imshow("img", img)
        cv2.waitKey()

        # 이미지를 784개 흑백 픽셀로 사이즈 변환
        img = img.reshape(-1, 28, 28, 1)

        # 데이터를 모델에 적용할 수 있도록 가공
        input_data = ((np.array(img) / 255) - 1) * -1

        # 클래스 예측 함수에 가공된 테스트 데이터 넣어 결과 도출
        res = np.argmax(model.predict(input_data), axis=-1)
        number = number + str(res[0])
    print(number)
    return(number)

# img = cv2.imread("750.jpg")
# digit_prediction(img)




