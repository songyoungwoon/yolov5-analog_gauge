# model.save("/content/mnist_model.h5")

import time
import datetime

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

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

def digit_prediction(img):
    # 이미지 읽어오기
    plt.figure(figsize=(15, 12))

    # 이미지 흑백처리
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img_gray, dsize=(120, 120), interpolation=cv2.INTER_AREA)

    model = tf.keras.models.load_model('C:/Users/syw/Desktop/yolov5/mnist_models/aaa.h5')

    # 이미지를 784개 흑백 픽셀로 사이즈 변환
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    input_data = img.reshape(1, 120,120,3)

    number = ""
    predictions = model.predict(input_data)
    score = tf.nn.softmax(predictions[0])
    number = number + str(class_names[np.argmax(score)])
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    return(number)


# for j in range(10):
#     for i in range(17, 21, 1):
#         img = cv2.imread(f"rect/{i}.jpg")
#         digit_prediction(img)
path = 'C:/Users/syw/Desktop/yolov5/runs/rect_imgs' # 폴더 경로
os.chdir(path) # 해당 폴더로 이동
files = os.listdir(path)

png_img = []
jpg_img = []
for file in files:
    if '.png' in file:
        f = cv2.imread(file)
        png_img.append(f)
    if '.jpg' in file:
        f = cv2.imread(file)
        jpg_img.append(f)

for img in jpg_img:
    digit_prediction(img)
    cv2.imshow("img", img)
    cv2.waitKey()



