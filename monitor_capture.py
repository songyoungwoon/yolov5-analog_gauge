from PIL import ImageGrab
import cv2
import numpy as np
import os
import datetime
import time

folder_name = 'classify_angle'
x1, y1, x2, y2 = 0, 0, 500, 500
while True:
    image = cv2.cvtColor(np.array(ImageGrab.grab(bbox=(x1, y1, x2, y2))), cv2.COLOR_BGR2RGB)
    # cv2.imshow("image", image)
    angle = 0
    if os.path.exists(os.path.join('./' + folder_name + '/' + str(angle))):
        now_time = str(datetime.datetime.now().strftime("%y%m%d_%H%M%S"))
        u = f"./{folder_name}/{str(angle)}/{now_time}.jpg"
        cv2.imwrite(u, image)
        print(f"save")
        time.sleep(12)
#   key = cv2.waitKey(100)
#   if key == ord("q"):
#       print("Quit")
#       break

cv2.destroyAllWindows()