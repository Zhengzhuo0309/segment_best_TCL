# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/1 18:54
@Auth ： pan.xie
@File ：capture_img_url.py
@IDE ：PyCharm
"""
import cv2
path = "rectify"
video = "2023_rectify.mp4"
cap = cv2.VideoCapture("{}/{}".format(path,video))
isOpened = cap.isOpened()# 判断是否打开‘
print("isOpened = ",isOpened)
fps = cap.get(cv2.CAP_PROP_FPS)# 帧率 每秒展示多少张图片
print("fps = ",fps)
i = 0
while(cap.isOpened()):
    print(i)
    i = i+1
    ret, frame = cap.read()
    # print("frame = ",frame)

    if frame is not None:
        frame = cv2.resize(frame, (1920,1080))
    else:
        break
    # print(frame.shape)
    # cv2.imshow('frame',frame)
    cv2.imwrite("./{}/{}.jpg".format(path,i), frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
