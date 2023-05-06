import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re


def pngfilter(f):
    if f[-4:] in ['.jpg', '.png', '.bmp']:
        return True
    else:
        return False
data_root = os.getcwd()+'/vis_img_test'
# save_root = os.getcwd()+'/vis_img_test'
img_file_list_with_sub = os.listdir(os.getcwd() + '/vis_img_test')
img_file_list_rand = list(filter(pngfilter, img_file_list_with_sub))
img_file_list = sorted(img_file_list_rand, key=lambda x: int(re.findall(r"\d+", x)[0]))
wait_time = 1
i = 1
for image_path in img_file_list:
    image_path_root = os.path.join(data_root, image_path)

    # 读取图像
    img = cv2.imread(image_path_root)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # # 转换为灰度图像
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # # 进行阈值分割
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #
    # # 进行形态学操作，去除噪声和小区域
    # kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    fig = plt.figure()
    info = plt.imshow(img, cmap='gray')
    plt.pause(wait_time)
    plt.show()
    # plt.savefig(os.path.join(save_root, image_path))
    print(i,info)
    i+=1



