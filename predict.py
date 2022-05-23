import tensorflow as tf
import cv2
import numpy as np
from utils.general import LOGGER

# 모델 불러오기
model = tf.keras.models.load_model('./proto1.h5')
crop = cv2.imread('1.jpg')
# 이미지 size 조정
# int -> float
scalingFactor = 1 / 255.0
# Convert unsigned int 8bit to float
crop = np.float32(crop)
crop = crop * scalingFactor

# input layer 형식 맞추기
crop = cv2.resize(crop, (120, 120))
crop = crop.reshape(1, 120, 120, 3)

# model 평가 및 예측값
a = np.squeeze(model.predict(crop))
LOGGER.info(f'predict : {a}')

