from contextInfo.context import Context
import cv2
import numpy as np
import json
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def rectify_video(image, map1, map2):
    frame_rectified = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return frame_rectified


def calculate_para(K, D, width, height):
    # 优化内参数和畸变系数
    p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), None)
    # 此处计算花费时间较大，需从循环中抽取出来
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, None, p, (width, height), cv2.CV_32F)
    return map1, map2

def current_coord(detect_result,results):
    pen_coords = [item for item in detect_result if item["name"] == "pen"]
    screen_coords = [item for item in detect_result if item["name"] == "screen"]
    sorted_pen = sorted(pen_coords, key=lambda x: x.get("confidence"),reverse=True)
    sorted_coord_screen = sorted(screen_coords, key=lambda x: x.get("confidence"), reverse=True)

    screen_rect = []
    pen_rect = []
    pen_point = [-1,-1]
    if sorted_coord_screen:
        screen_x1 = int(sorted_coord_screen[0]["xmin"])-10
        screen_y1 = int(sorted_coord_screen[0]["ymin"])-10
        screen_x2 = int(sorted_coord_screen[0]["xmax"])+5
        screen_y2 = int(sorted_coord_screen[0]["ymax"])+5
        screen_rect = [screen_x1,screen_y1,screen_x2,screen_y2]

    if sorted_pen:
        # print("sort_pen = ", sorted_pen)
        # results.show()  # 显示标注
        pen_x1 = int(sorted_pen[0]["xmin"])
        pen_y1 = int(sorted_pen[0]["ymin"])
        pen_x2 = int(sorted_pen[0]["xmax"])
        pen_y2 = int(sorted_pen[0]["ymax"])
        pen_point = [pen_x2,pen_y2]
        pen_rect = [pen_x1,pen_y1,pen_x2,pen_y2]
    return pen_point,pen_rect,screen_rect

class edgeDetectBase(object):

    def __init__(self):
        self.SN = "002"
        self.area_per_min = 0
        self.box_all = []
        self.count_keep_max=3
        self.dist_threshold =3
        self.count_keep = 0
        self.area_per_change_threshold = 0.005
        pass

    def edge_handle(self,img,screen_rect):
        # 屏幕 截取
        if screen_rect:
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.rectangle(mask, (screen_rect[0], screen_rect[1]), (screen_rect[2], screen_rect[3]), 255, -1)

            screen_masked = cv2.bitwise_and(img, img, mask=mask)
            # cv2.imshow("rectangular mask", masked)
        else:
            screen_masked = copy.deepcopy(img)

        img_handle = self.color_select(screen_masked)

        seg_img = np.zeros_like(img)
        img_copy = copy.deepcopy(img)
        # 转换为灰度图像
        gray = cv2.cvtColor(img_handle, cv2.COLOR_BGR2GRAY)

        gray_copy = copy.deepcopy(gray)

        # 对灰度图形态学处理 返回 mask
        mask, closed_mask, mask_er = self.egde_detect_biology(img,gray_copy)

        gray_max, gray_mask_max, gray_mask_max_bool = self.find_mask_max(closed_mask, gray_copy)

        gray, gray_mask = self.rect_mask_filter(gray_mask_max, gray_max)

        kernel_5x5 = np.ones((5, 5), np.uint8)
        gray_mask_ode = cv2.erode(gray_mask_max, kernel_5x5, iterations=50)
        # cv2.imshow("closed_mask",gray_mask_ode)
        # 遮挡后，最大区域为外边框
        screen_contours = self.findcountours(gray_mask_ode)
        # screen_contours_old.append(screen_contours)

        #外接矩形
        rect = cv2.minAreaRect(screen_contours[0])
        # rect_all.append(rect)
        box = cv2.boxPoints(rect)
        box = np.int64(box)
        #
        self.box_all.append(box)   #没有经过排序
        #
        cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)
        # cv2.imshow("xx",img_copy)
        #
        area = cv2.contourArea(box)
        area_contour = cv2.contourArea(screen_contours[0])
        area_per = area_contour/area#面积比例

        # # 边框迭代优化
        # # -------面积比变大暂存---------
        # if (self.area_per_min < area_per):
        #     screen_contours_best_tmp = screen_contours
        #     area_per_min_tmp = area_per
        #     box_tmp = box
        # # -------面积比变大暂存---------
        # #1. area_per_min > area_per_cur  #留下 area_per_min = area_per_cur
        # # 2.角点变化  #新SN area_per_min = area_per_cur
        # if len(self.box_all)>self.count_keep_max:
        #
        #     box_past = self.box_all[-2]   #上一次识别框外接矩形
        #
        #     #判断识别框是否长时间不变
        #     M_past = cv2.moments(box_past)
        #     cx_past = int(M_past['m10'] / M_past['m00'])
        #     cy_past = int(M_past['m01'] / M_past['m00'])
        #     point_past = np.array([cx_past, cy_past])
        #     M = cv2.moments(box)
        #     cx = int(M['m10'] / M['m00'])
        #     cy = int(M['m01'] / M['m00'])
        #     point = np.array([cx, cy])
        #     dist = np.linalg.norm(point_past-point)      #重心变化
        #     #-------重心保持---------
        #     if (dist < self.dist_threshold):
        #         self.count_keep += 1
        #     else:
        #         self.count_keep = 0
        #     # -------重心保持---------
        #
        #     # -------重心保持暂存---------
        #     if (self.count_keep >= self.count_keep_max):
        #         screen_contours_best_tmp = screen_contours
        #         box_tmp = box
        #         area_per_min_tmp = area_per
        #     # -------重心保持---------
        #
        #     # -------处理初始帧-----
        # else:
        #     screen_contours_best = screen_contours_best_tmp
        #     area_per_min = area_per_min_tmp
        #     box_best = box_tmp
        #     # -------处理初始三帧-----
        #
        # # -------重心保持覆盖---------   条件：面积比变小程度不大，重心变化不大，并且保持了一段时间
        # if (dist < self.dist_threshold) and (self.count_keep >= self.count_keep_max) and ((area_per_min - area_per) < area_per_change_threshold):
        #
        #     screen_contours_best = screen_contours_best_tmp
        #     box_best = box_tmp
        #     area_per_min = area_per_min_tmp
        #     count_keep = 0
        # # -------重心保持覆盖---------
        #
        # area_per_list.append(area_per)
        # area_per_min_list.append(area_per_min)
        #
        # #box排序，便于后面显示可用区域
        # box_best_xs = [i[0] for i in box_best]
        # box_best_ys = [i[1] for i in box_best]
        # box_best_xs.sort()
        # box_best_ys.sort()
        # box_best = np.array([[box_best_xs[1],box_best_ys[1]],[box_best_xs[2],box_best_ys[1]],
        #                    [box_best_xs[2],box_best_ys[2]],[box_best_xs[1],box_best_ys[2]]])
        #
        # #最比例最大的（找到最好的框）---->  框长时间不变，开启角点检测  ----> 角点变化，当做新的SN检测，但是保留轨迹
        # screen_contours_new.append(screen_contours_best)
        # box_all_new.append(box_best)  #经过排序后
        #
        # ##是否需要新的外接矩形
        # # 创建新的图像并绘制分割轮廓
        # img_test = np.zeros_like(img_copy)
        # cv2.drawContours(img_test, screen_contours_best, -1, (0, 0, 255), 2)
        # cv2.drawContours(img_copy, screen_contours_best, -1, (0, 0, 255), 2)
        # # cv2.drawContours(img_copy, [box_best], -1, (255, 0, 0), 2)
        # # gray_copy[closed_mask == 255] = 255
        #
        # #内接矩形
        # rect_in = order_points(screen_contours_best[0].reshape(screen_contours_best[0].shape[0], 2))
        # rect_in = np.int64(rect_in)
        # xs_in = [i[0] for i in rect_in]
        # ys_in = [i[1] for i in rect_in]
        # xs_in.sort()
        # ys_in.sort()
        # # 内接矩形的坐标为
        # # print(xs_in[1], xs_in[2], ys_in[1], ys_in[2])
        # box_in = np.array([[xs_in[1],ys_in[1]],[xs_in[2],ys_in[1]],
        #                    [xs_in[2],ys_in[2]],[xs_in[1],ys_in[2]]])
        # box_in_buffer = np.array([[xs_in[1]+buffer+buffer_in_right,ys_in[1]+buffer],[xs_in[2]-buffer,ys_in[1]+buffer],
        #                           [xs_in[2]-buffer,ys_in[2]-buffer],[xs_in[1]+buffer+buffer_in_right,ys_in[2]-buffer]])
        # # cv2.drawContours(img_copy, [box_in], -1, (0, 255, 255), 2)
        # x_inbuffer_left, x_inbuffer_right, y_inbuffer_up, y_inbuffer_down = box_in_buffer[0][0], box_in_buffer[1][0], \
        #                                                            box_in_buffer[1][1], box_in_buffer[2][1]
        # #有效区域
        # box_buffer = np.array([[box_best[0][0] - buffer, box_best[0][1] - buffer], [box_best[1][0] + buffer+ buffer_right, box_best[1][1] - buffer],
        #                        [box_best[2][0] + buffer + buffer_right, box_best[2][1] + buffer], [box_best[3][0] - buffer, box_best[3][1] + buffer]])
        # xbuffer_left, xbuffer_right, ybuffer_up, ybuffer_down = box_buffer[0][0], box_buffer[1][0], \
        #                                                            box_buffer[1][1], box_buffer[2][1]
        #
        # rectangle_out_mask = np.zeros(img.shape[0:2], dtype="uint8")
        # cv2.rectangle(rectangle_out_mask, tuple(box_buffer[0]), tuple(box_buffer[2]), 255, -1)
        # rectangle_in_mask = np.zeros(img.shape[0:2], dtype="uint8")
        # cv2.rectangle(rectangle_in_mask, tuple(box_in_buffer[0]), tuple(box_in_buffer[2]), 255, -1)
        # buffer_area = rectangle_out_mask-rectangle_in_mask
        # buffer_area_bool = (buffer_area==255)
        # area_zeros1 = 255-np.zeros_like(buffer_area)
        # area_zeros1[buffer_area_bool] = 0
        # area_zeros2 = 255-np.zeros_like(buffer_area)
        # area_zeros2[buffer_area_bool] = 0
        # area_zeros3 = 255 - np.zeros_like(buffer_area)
        # buffer_area_img = np.dstack((area_zeros1, area_zeros3, area_zeros2))

    @staticmethod
    def color_select(img):
        """
        阈值过滤 过滤掉多余颜色像素
        :param img:
        :return:
        """
        blur = cv2.blur(img, (5, 5))
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # low_blue = np.array([100,100,100])
        # high_blue = np.array([180,180,180])
        low_blue = np.array([10, 2, 10])
        high_blue = np.array([220, 32, 230])
        # 可实现二值化功能（类似threshold()函数）可以同时针对多通道进行操作
        mask = cv2.inRange(hsv, low_blue, high_blue)
        res = cv2.bitwise_and(img, img, mask=mask)
        return res

    @staticmethod
    def egde_detect_biology(img,gray):
        """
        形态学处理
        :param gray: 灰度图
        :return: maskq
        """
        # 进行阈值分割

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(gray, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=int(1080 / 3 - 1), C=20)
        # thresh = cv2.adaptiveThreshold(gray, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=9, C=5)

        cv2.imshow("closed_mask", thresh)
        # 进行形态学操作，去除噪声和小区域
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算
        # 进行连通区域分析
        cv2.imshow("closed_mask2", opening)

        # canny边缘
        canny_edge = cv2.Canny(opening,128,255)

        cv2.imshow("canny",canny_edge)

        # 找出轮廓
        # contours, hierarchy = cv2.findContours(opening, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        #
        # for cnt in contours:
        #     if len(cnt)>4:
        #         # rect = cv2.minAreaRect(cnt)
        #         # box = cv2.boxPoints(rect)
        #         # box = np.int0(box)q
        #         # cv2.drawContours(img,[box],0,(0,0,255),2)
        #         epsilon = 0.001 * cv2.arcLength(cnt, True)
        #         approx = cv2.approxPolyDP(cnt, epsilon, True)
        #         cv2.drawContours(img, approx, -1, (0, 255, 0), 5)
        #         cv2.polylines(img, [approx], True, (0, 0, 255), 2)

        lines = cv2.HoughLinesP(canny_edge, 1, np.pi / 180, 100, minLineLength=120, maxLineGap=50)

        # 笔在哪一边时候，那边的边不确定
        up = []
        down = []
        left = []
        right = []
        coords = []
        for i,line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            if abs(y2-y1)>abs(x2-x1):
                # y轴方向 y轴变化大
                if x1<960 and x2<960:
                    coord = "left"
                    left.append(line[0])
                elif x1>960 and x2>960:
                    coord = "right"
                    right.append(line[0])
                else:
                    coord = ""
            else:
                # x轴方向 x轴变化大
                if y1<540 and y2<540:
                    coord = "up"
                    up.append(line[0])
                elif y1>540 and y2>540:
                    coord = "down"
                    down.append(line[0])
                else:
                    coord = ""
            coords.append(coord)


        print(coords)
        for line in lines:
            print("lines = ",line)
            # 获取坐标
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)
        cv2.imshow("imag", img)





        n, labels, stats, centroids = cv2.connectedComponentsWithStats(opening)
        area_label_list = list(np.where((stats[:, 4] < 1000000) & (stats[:, 4] > 100000))[0])
        # #找到最大连通区域
        # max_area = 0
        # max_label = 0
        # for i in range(0, n):
        #     if stats[i, cv2.CC_STAT_AREA] > max_area:
        #         max_area = stats[i, cv2.CC_STAT_AREA]
        #         max_label = i
        # stats[np.argsort(-stats[:, 4])][1:3, :]
        # # 提取最大连通区域
        # mask = np.zeros_like(gray)
        # mask[labels == max_label] = 255

        # 提取最大连通区域
        mask = np.zeros_like(gray)
        for area_label in area_label_list:
            mask[labels == area_label] = 255
        cv2.imshow("mask", mask)
        # 进行形态学操作，填充区域
        kernel_5x5 = np.ones((5, 5), np.uint8)
        kernel_3x3 = np.ones((3, 3), np.uint8)
        mask_er = cv2.dilate(mask, kernel_5x5, iterations=50)
        closed_mask = cv2.morphologyEx(mask_er, cv2.MORPH_CLOSE, kernel_5x5)

        # # 进行轮廓检测
        # contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return mask, closed_mask, mask_er
    @staticmethod
    def getLinearEquation(p1x, p1y, p2x, p2y):
        sign = 1
        a = p2y - p1y
        if a < 0:
            sign = -1
            a = sign * a
        b = sign * (p1x - p2x)
        c = sign * (p1y * p2x - p1x * p2y)
        return [a, b, c]


    # mask最大区域
    @staticmethod
    def find_mask_max(closed_mask, gray):
        gray_copy = copy.deepcopy(gray)
        y, x = gray_copy.shape
        img_center_x = np.int32(x / 2)
        img_center_y = np.int32(y / 2)
        img_center_point = np.array([img_center_x, img_center_y])
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - closed_mask, connectivity=4)
        centroids_list = centroids.astype(np.int32)
        diff_dist = np.expand_dims(np.linalg.norm(img_center_point - centroids_list, axis=1), axis=1)
        diff_dist_min = diff_dist[1:].min() # 距离最小
        max_label = np.where(diff_dist == diff_dist_min)[0][0]
        # max_label = np.where(stats[:,4]==stats[1:,4].max())[0][0]
        max_label = max_label.astype(np.int32)
        gray_mask_bool = ~(labels == max_label)
        gray_mask = np.zeros_like(gray)
        gray_mask[gray_mask_bool] = 255
        gray_copy[gray_mask_bool] = 255
        return gray_copy, gray_mask, gray_mask_bool

    # 过滤矩形外区域
    @staticmethod
    def rect_mask_filter(gray_mask, gray):
        gray_mask_copy = copy.deepcopy(gray_mask)
        gray_copy = copy.deepcopy(gray)
        mask_num = np.sum(gray_mask == 0, 1)
        di = np.diff(mask_num)
        di = np.append(np.zeros(1, int), di)
        y_up = np.where(di == di.max())[0][0]
        y_down = np.where(di == di.min())[0][0]
        gray_copy[:y_up + 1, :] = 255
        gray_copy[y_down:, :] = 255
        gray_mask_copy[:y_up + 1, :] = 255
        gray_mask_copy[y_down:, :] = 255
        # plt.figure(1)
        # plt.subplot(1,2,1)
        # plt.plot(list(range(len(mask_num))),mask_num)
        # plt.subplot(1,2,2)
        # plt.plot(list(range(len(di))),di)
        # plt.show()
        return gray_copy, gray_mask_copy
    @staticmethod
    def findcountours(gray_mask_ode):
        contours, hierarchy = cv2.findContours(255 - gray_mask_ode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        screen_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > 1 and aspect_ratio < 5:
                screen_contours.append(contour)
        return tuple(screen_contours)
    def drop_context(self):
        pass

def main_logic(scan_info,model):
    # 根据扫描信息判断电视版信息
    # 根据扫描信息判断板子是否为同一个，为同一个则不清空信息
    # 不同的则清空给上下文信息
    # 打开视频流
    # 获取边框
    # 获取检测
    # 路径分析判断
    # 结果控制设备
    # 开始时候灯由熄灭状态 -> 黄色
    # 操作完成  黄色 -> 绿色 -> 熄灭

    # video = cv2.VideoCapture(0)
    # video = cv2.VideoCapture("./data/20230228_4mm.mp4")
    video = cv2.VideoCapture("./data/0314_left.mp4")
    ID = Context.getID()
    print("ID = ",ID)
    i = 0

    Context.change(scan_info)

    ID = Context.getID()
    print("ID new = ", ID)
    K = np.array(
        [[1201.3967181633418, 0.0, 909.7424436183744], [0.0, 1203.635467250557, 534.1590658991514], [0.0, 0.0, 1.0]])
    D = np.array([[-0.0978537375125563], [-0.03841501213366177], [-0.03612764818273854], [0.05276041355808103]])
    width, height = 1920, 1080
    # width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # print(width, height)
    map1,map2 = calculate_para(K,D,width,height)
    edgeDetect = edgeDetectBase()
    while (video.isOpened()):
        """
        0、矫正
        1、获取检测框，检测框缓存
        2、获取检测点
        3、判断检测区域
        4、决策逻辑 
        5、设备控制
        """
        success, image = video.read()
        if not success:
            break

        # cv2.imwrite("./data2/{}.jpg".format(i), image)
        i = i + 1

        # 图片矫正
        frame = rectify_video(image, map1, map2)

        # frame = cv2.resize(frame, (960, 540))
        # cv2.imshow('frame', frame)

        # 边框检测

        # 算法模型
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        detect_result = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
        # for box in detect_result:
        #     l, t, r, b = int(box["xmin"]),int(box["ymin"]),int(box["xmax"]),int(box["ymax"])
        #     confidence = round(box["confidence"],1)
        #     cls_name = box["name"]
        #     # outside = t - h >= 3
        #     if cls_name=="pen" or cls_name=="screen":
        #         cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        #         cv2.putText(frame, cls_name + "-" + str(confidence), (l, t), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        # cv2.imshow("ff",frame)

        # 当前笔的坐标,笔的位置，屏幕位置
        pen_point,pen_rect,screen_rect = current_coord(detect_result,results)

        # 边框提取
        edgeDetect.edge_handle(frame,screen_rect)

        #




        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
        # Context.add(i)
        # if i%2==1:
        #     Context.sub(i)
        # if i>100:
        #     break

        # 决策分析


    video.release()

    cv2.destroyAllWindows()
    return {"i":i,"ID":ID}

if __name__=="__main__":
    import torch

    model = torch.hub.load("D:\project\demo", "custom", 'D:\project\demo\model_save\ybest_screen', source="local")
    model.conf = 0.6  # NMS confidence threshold
    model.iou = 0.2
    main_logic(1,model)

    # A082
