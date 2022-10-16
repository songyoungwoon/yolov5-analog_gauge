import glob
import os

def remove():
    path = "./classify_angle (3)/classify_angle/Train/"
    # path = "./360/"
    for i in range(360):
        temp_path = path+str(i)
        pic = glob.glob(temp_path+"\\*")
        os.remove(pic[0])
remove()