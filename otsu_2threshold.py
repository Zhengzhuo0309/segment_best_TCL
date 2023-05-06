import numpy as np
def Otsu2Threshold(src):
    Threshold1 = 0
    Threshold2 = 0
    theta = 0
    weight, height = np.shape(src)
    hest = np.zeros([256], dtype=np.int32)
    for row in range(weight):
        for col in range(height):
            pv = src[row, col]
            hest[pv] += 1
    tempg = -1
    N_blackground = 0
    N_object = 0
    N_all = weight * height
    for i in range(256):
        N_object += hest[i]
        for k in range(i, 256, 1):
            N_blackground += hest[k]
        for j in range(i, 256, 1):
            gSum_object = 0
            gSum_middle = 0
            gSum_blackground = 0

            N_middle = N_all - N_object - N_blackground
            w0 = N_object / N_all
            w2 = N_blackground / N_all
            w1 = 1 - w0 - w2
            for k in range(i):
                gSum_object += k*hest[k]
            u0 = gSum_object/N_object
            for k in range(i+1, j, 1):
                gSum_middle += k*hest[k]
            u1 = gSum_middle / (N_middle+theta)

            for k in range(j+1, 256, 1):
                gSum_blackground += k*hest[k]
            u2 = gSum_blackground / (N_blackground + theta)

            u = w0 * u0 + w1 * u1 + w2 * u2
            # print(u)
            g = w0 * (u - u0) * (u - u0) + w1 * (u - u1) * (u - u1) + w2 * (u - u2) * (u - u2)
            if tempg < g:
                tempg = g
                Threshold1 = i
                Threshold2 = j
            N_blackground -= hest[j]

    h, w = np.shape(src)
    img = np.zeros([h, w], np.uint8)
    for row in range(h):
        for col in range(w):
            if src[row, col] > Threshold2:
                img[row, col] = 255
            elif src[row, col] <= Threshold1:
                img[row, col] = 0
            else:
                img[row, col] = 126
    BlackgroundNum = 0
    AllNum = weight*height
    for i in range(weight):
        for j in range(height):
             if img[i, j] == 0:
                BlackgroundNum += 1
    BlackgroundRatio = BlackgroundNum/AllNum
    if BlackgroundRatio < 0.4: # 背景占比过少时，做一个反向操作
        w, h = np.shape(src)
        for i in range(w):
            for j in range(h):
                img[i, j] = 255 - img[i, j]
    return img