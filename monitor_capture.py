from PIL import ImageGrab
import cv2
import numpy as np
import os
import datetime
import time

angle = 0
folder_name = 'classify_angle'
# for i in range(0, 360):
#     if not os.path.exists(
#             os.path.join('./' + folder_name + '/' + str(i))):  # make 0 ~ 360 directory
#         os.mkdir('./' + folder_name + '/' + str(i))
#left, up, right, down
x1, y1, x2, y2 = 335, 200, 885, 745 #clock_1
while True:
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2))), cv2.COLOR_BGR2RGB)
    if os.path.exists(os.path.join('./' + folder_name + '/' + str(angle))):
        now_time = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
        u = f"./{folder_name}/{str(angle)}/{now_time}.jpg"
        cv2.imwrite(u, image)
        print(f"save")
        time.sleep(9.95)
        angle += 1

    # cv2.imshow("image", image)
    # key = cv2.waitKey(100)
    # if key == ord("q"):
    #     print("Quit")
    #     break

# cv2.destroyAllWindows()