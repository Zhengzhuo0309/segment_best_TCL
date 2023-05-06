from contextInfo.context import Context
import cv2
import numpy as np
import json
import copy
import numpy as np
import math
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
    pen_point_center = [-1,-1]
    pen_center =[]
    if sorted_coord_screen:
        screen_x1 = int(sorted_coord_screen[0]["xmin"])-20
        screen_y1 = int(sorted_coord_screen[0]["ymin"])-20
        screen_x2 = int(sorted_coord_screen[0]["xmax"])+20
        screen_y2 = int(sorted_coord_screen[0]["ymax"])+20
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
        pen_center = [int((sorted_pen[0]["xmin"]+sorted_pen[0]["xmax"])/2),int((sorted_pen[0]["ymin"]+sorted_pen[0]["ymax"])/2)]
    return pen_point,pen_center,pen_rect,screen_rect

class edgeDetectBase(object):

    def __init__(self):
        self.SN = "002"
        self.area_per_min = 0
        self.box_all = []
        self.count_keep_max=3
        self.dist_threshold =3
        self.area_per_change_threshold = 0.005


        self.count_pic = 0
        self.count_keep = 0
        self.count_move = 0
        self.screen_move_keep_state = 0 # 0 keep 1 move  默认保持静止，针对第一帧
        self.max_count_keep = 3
        self.max_count_move = 3
        self.history_lines_left = []
        self.history_lines_up = []
        self.history_line_down = []
        self.history_lines_right = []
        # 历史点记录
        self.history_point_left_up = []
        self.history_point_left_down = []
        self.history_point_right_up = []
        self.history_point_right_dwon = []
        self.context_left_up = None
        self.context_left_down = None
        self.context_right_up = None
        self.context_right_down = None


    def rest(self):
        pass
    def edge_handle(self,img,screen_rect,pen_point):
        # 根据识别的屏幕边框 遮盖原图
        screen_masked = self.screen_mask(img,screen_rect)
        # 笔在屏幕的那个区域
        # cv2.imshow("screen_maske = ,",screen_masked)
        # 根据色域选择图
        color_masked = self.color_select(screen_masked)
        # cv2.imshow("color_masked ", color_masked)
        # 返回笔的位置
        location = self.judge_location_pen(pen_point, screen_rect)
        print("location = ", location)
        seg_img = np.zeros_like(img)
        img_copy = copy.deepcopy(img)
        # 转换为灰度图像
        gray = cv2.cvtColor(color_masked, cv2.COLOR_BGR2GRAY)
        # 获取轮廓的线段 ，提取轮廓的点
        lines = self.get_max_contours_lines(img,gray)

        # 每个边框的直线点
        line_points = self.judge_location_line(lines)

        # 四个边角点的坐标,四个边的直线
        predict_points,predict_lines = self.get_line_func(line_points,screen_rect)

        # 分析上下文
        current_scrrent_state,screen_move_keep_state,keep_state_begin,context_predict_points,context_predict_lines_points = self.context_alaysis(location,predict_points) # 当前面板的状态


        # 平移区域   可以缓存直线 减少运行时间

        predict_points_out,predict_points_in = self.get_line_func_move(context_predict_lines_points,screen_rect)

        for item in predict_points:
            cv2.circle(img, (item[0],item[1]), 3, (255, 255, 0), 3)
        for item in context_predict_points:
            cv2.circle(img, (item[0],item[1]), 3, (0, 0, 255), 3)
        for i,item in enumerate(predict_points_out):
            k = i +1 if i!=3 else 0
            cv2.line(img, tuple(predict_points_out[i]),tuple(predict_points_out[k]),  (0, 255, 255), thickness=2)
        for i, item in enumerate(predict_points_in):
            k = i + 1 if i != 3 else 0
            cv2.line(img, tuple(predict_points_in[i]), tuple(predict_points_in[k]), (0, 255, 255), thickness=2)
        cv2.imshow("xxxx",img)
        # predict_points = [[item] for item in predict_points]
        print("predict_points = ",predict_points)
        # 内外接矩形
        box_out = self.find_rectangle_out( predict_points) # 外接
        box_in = self.find_rectangle_in(predict_points) # 内接

        # 直线平移
        # print(box_out)
        # print(box_in,type(box_in))
        cv2.rectangle(img, (box_out[0],box_out[1]), (box_out[0]+box_out[2],(box_out[1]+box_out[3])), (255, 255, 0), 2)
        cv2.rectangle(img, tuple(box_in[0]), tuple(box_in[2]), (255,255,0), 2)
        cv2.imshow("box",img)
        # 当前状态 开始变换的时刻
        return screen_move_keep_state,keep_state_begin,predict_points_out,predict_points_in

    @staticmethod
    def reverse_black_white(pic):
        reverse_pic = 255-pic
        #
        # where_0 = np.where(pic==0)
        # where_255 = np.where(pic==255)
        # reverse_pic[where_255]=0
        # reverse_pic[where_0]=255
        return reverse_pic
    @staticmethod
    def two_point_distance(point1,point2):
        distance = math.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))
        print("point = ",point1,point2)
        print("distance = ",distance)
        return distance
    @staticmethod
    def find_rectangle_out(predict_points):
        # temp_image = np.zeros(gray.shape, np.uint8)
        # 绘制多边形
        pts = np.array(predict_points)
        # cv2.polylines(temp_image, [pts], True, (255, 255, 255), 5)
        # cv2.drawContours(temp_image, [pts], -1, (255, 255, 255), thickness=-1)
        # cv2.imshow("temp",temp_image)
        rect = cv2.boundingRect(pts)
        return rect
    @staticmethod
    def find_rectangle_in2(pts):
        # pts为轮廓坐标
        # 列表中存储元素分别为左上角，右上角，右下角和左下角
        pts = np.array(pts)
        # rect = np.zeros((4, 2), dtype="float32")
        rect = [[] for i in range(4)]
        # 左上角的点具有最小的和，而右下角的点具有最大的和
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)].tolist()
        rect[2] = pts[np.argmax(s)].tolist()
        # 计算点之间的差值
        # 右上角的点具有最小的差值,
        # 左下角的点具有最大的差值
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)].tolist()
        rect[3] = pts[np.argmax(diff)].tolist()
        # 返回排序坐标(依次为左上右上右下左下)
        return rect
    @staticmethod
    def find_rectangle_in(pts):
        up = max(pts[0][1],pts[1][1])
        right = min(pts[1][0],pts[2][0])
        down = min(pts[2][1],pts[3][1])
        left = max(pts[3][0],pts[0][0])
        rect = [[left,up],[right,up],[right,down],[left,down]]
        return rect

    @staticmethod
    def judge_location_line(lines):
        up = []
        down = []
        left = []
        right = []
        coords = []
        for i,line in enumerate(lines):
            # print("line = ",line)

            x1, y1, x2, y2 = line[0]
            point1 = [x1,y1]
            point2 = [x2,y2]
            if abs(y2-y1)>abs(x2-x1):
                # y轴方向 y轴变化大
                if x1<960 and x2<960:
                    coord = "left"
                    left.append(point1)
                    left.append(point2)
                elif x1>960 and x2>960:
                    coord = "right"
                    right.append(point1)
                    right.append(point2)
                else:
                    coord = ""
            else:
                # x轴方向 x轴变化大
                if y1<540 and y2<540:
                    coord = "up"
                    up.append(point1)
                    up.append(point2)
                elif y1>540 and y2>540:
                    coord = "down"
                    down.append(point1)
                    down.append(point2)
                else:
                    coord = ""
            coords.append(coord)
        return [up,down,left,right]
    def context_alaysis(self,location,predict_points):
        """
        在左边：右边的点稳定
        在右边：左边的点稳定
        其他位置：左右边都稳定
        1、记录每个点的位置。并计算当前点位置与上一位置的偏移量，每次计算两个坐标点，在左边计算右边点，在右边计算左边点

        先判断边框是否挪动，在左边，判断右边两点，在右边，判断左边两点
        边框没动时
            在右边，右边点可以取历史右边稳定点。 左边可取历史左边稳定点
        边框挪动时(在左边时，右边的点有变化，连续三次变化。)

        """

        left_up_point, right_up_point, right_down_point, left_down_point = predict_points # 当前的坐标
        current_scrrent_state = 0  # 当前单帧状态 0 静止 1 移动
        keep_state_begin = False
        # 分析当前状态
        if self.history_point_left_up: # 有历史数据
            if location == "right":
                # 在右边 左边的点相对稳定, 右边的点选取右边的历史点，右边的历史点，选取时，不能在location=right时选取。
                # right_up_point = self.context_right_up
                # 没有历史坐标点，或者历史坐标点移动小于某个阈值 证明保持稳定
                if self.two_point_distance(left_up_point, self.history_point_left_up[-1])<10 and self.two_point_distance(left_down_point,self.history_point_left_down[-1])<10:
                    pass
                else:
                    current_scrrent_state=1
            elif location == "left":
                if self.two_point_distance(right_up_point, self.history_point_right_up[-1])<10 and self.two_point_distance(right_down_point,self.history_point_right_dwon[-1])<10:
                    pass
                else:
                    current_scrrent_state=1
            else:
                if self.two_point_distance(left_up_point, self.history_point_left_up[-1])<10 and self.two_point_distance(left_down_point,self.history_point_left_down[-1])<10:
                    pass
                else:
                    current_scrrent_state = 1
        else:
            # 没有历史数据，当作当前静止，主要针对第一帧 第一帧记录历史边框记录
            pass

        # 当前状态为移动时，判断移动的次数，移动的次数超过max时，将静止置为空
        if current_scrrent_state == 1:
            self.count_move +=1 # 移动加1
            if self.count_move>self.max_count_move: # 移动的次数超过最大次数，静止置空
                self.count_keep = 0 # 静止置空
                self.screen_move_keep_state = 1 # 修改当前状态为运动
        # 当前状态为静止时，判断移动的次数，移动的次数超过max时，将移动置为空
        else:
            self.count_keep += 1 # 静止加1
            if self.count_keep>self.max_count_keep: # 静止次数超过最大次数，移动置空
                self.count_move = 0 # 移动置空
                self.screen_move_keep_state = 0  # 修改当前状态为静止
        print("当前状态和屏幕上下文状态 = ",current_scrrent_state,self.screen_move_keep_state,self.count_move,self.count_keep)


        # 修改上下文的坐标信息
        if not self.history_point_left_up:
            keep_state_begin = True
            self.context_left_up = left_up_point
            self.context_left_down = left_down_point
            self.context_right_down = right_down_point
            self.context_right_up = right_up_point
        else:
            # 保持次数等于最大次数时候 该时刻已经从运动状态切换到静止状态
            # 修改上下文的当前坐标点信息，运动时不修改坐标点信息
            if self.count_keep == self.max_count_keep and len(self.history_point_left_up)!=self.max_count_keep: # 移动后到修改的临界点
                keep_state_begin = True
                if location == "right":
                    self.context_left_up = left_up_point
                    self.context_left_down = left_down_point
                elif location == "left":
                    self.context_right_down = right_down_point
                    self.context_right_up = right_up_point
                else:
                    self.context_left_up = left_up_point
                    self.context_left_down = left_down_point
                    self.context_right_down = right_down_point
                    self.context_right_up = right_up_point

        # 修改后的当前坐标信息
        # 当前坐标信息，为修改后的上下文坐标，可直接返回

        # 保存历史状态
        self.history_point_left_up.append(left_up_point)
        self.history_point_right_up.append(right_up_point)
        self.history_point_right_dwon.append(right_down_point)
        self.history_point_left_down.append(left_down_point)
        # 当前状态  上下文状态  移动后稳定的起始点  左上 右上 右下 左下 ，上下左右边的信息
        context_points = [self.context_left_up,self.context_right_up,self.context_right_down,self.context_left_down]
        # 上 下 左 右
        context_line_points = [[self.context_left_up,self.context_right_up],[self.context_right_down,self.context_left_down],[self.context_left_down,self.context_left_up],[self.context_right_up,self.context_right_down]]
        return current_scrrent_state,self.screen_move_keep_state,keep_state_begin,context_points,context_line_points





    def get_line_func(self,points,screen_rect):
        # 得到每个边界的曲线
        line_up = self.fit_line(points[0]) if points[0] else [0,1,-screen_rect[1]] # 上
        line_down = self.fit_line(points[1]) if points[1] else [0,1,-screen_rect[3]] # 下
        line_left = self.fit_line(points[2]) if points[2] else [1,0,-screen_rect[0]] # 左
        line_right = self.fit_line(points[3]) if points[3] else [1,0,-screen_rect[2]] # 左
        # 取得相交的点
        left_up_point = self.getLineCrossPoint(line_up,line_left)
        left_down_point = self.getLineCrossPoint(line_left,line_down)
        right_up_point = self.getLineCrossPoint(line_right,line_up)
        right_down_point = self.getLineCrossPoint(line_right,line_down)
        # print(left_up_point,left_down_point,right_up_point,right_down_point)
        return [left_up_point,right_up_point,right_down_point,left_down_point],[line_up,line_right,line_down,line_left]
    def get_line_func_move(self,points,screen_rect):
        buffer_out = 30
        buffer_in = 30
        line_up = self.fit_line(points[0]) if points[0] else [0,1,-screen_rect[1]] # 上
        line_down = self.fit_line(points[1]) if points[1] else [0,1,-screen_rect[3]] # 下
        line_left = self.fit_line(points[2]) if points[2] else [1,0,-screen_rect[0]] # 左
        line_right = self.fit_line(points[3]) if points[3] else [1,0,-screen_rect[2]] # 左

        line_up_out = [line_up[0],line_up[1],line_up[2]+line_up[1]*buffer_out]
        line_down_out = [line_down[0],line_down[1],line_down[2]-line_down[1]*buffer_out]
        line_left_out = [line_left[0],line_left[1],line_left[2]+line_left[0]*buffer_out]
        line_right_out = [line_right[0],line_right[1],line_right[2]-line_right[0]*buffer_out]

        line_up_in = [line_up[0],line_up[1],line_up[2]-line_up[1]*buffer_out]
        line_down_in = [line_down[0],line_down[1],line_down[2]+line_down[1]*buffer_out]
        line_left_in = [line_left[0],line_left[1],line_left[2]-line_left[0]*buffer_out]
        line_right_in = [line_right[0],line_right[1],line_right[2]+line_right[0]*buffer_out]

        left_up_point_out = self.getLineCrossPoint(line_up_out,line_left_out)
        left_down_point_out = self.getLineCrossPoint(line_left_out,line_down_out)
        right_up_point_out = self.getLineCrossPoint(line_right_out,line_up_out)
        right_down_point_out = self.getLineCrossPoint(line_right_out,line_down_out)

        left_up_point_in = self.getLineCrossPoint(line_up_in,line_left_in)
        left_down_point_in = self.getLineCrossPoint(line_left_in,line_down_in)
        right_up_point_in = self.getLineCrossPoint(line_right_in,line_up_in)
        right_down_point_in = self.getLineCrossPoint(line_right_in,line_down_in)
        # print(left_up_point,left_down_point,right_up_point,right_down_point)
        return [left_up_point_out,right_up_point_out,right_down_point_out,left_down_point_out],[left_up_point_in,right_up_point_in,right_down_point_in,left_down_point_in]
    def fit_line(self,point):
        line_up = cv2.fitLine(np.array(point), cv2.DIST_L2, 0, 0.01, 0.01)
        k = line_up[1] / line_up[0]
        b = line_up[3] - k * line_up[2]
        aa = round(k[0],2)
        bb = -1
        cc = round(b[0],2)
        return [aa,bb,cc]
    @staticmethod
    def getLineCrossPoint(line1, line2):
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        D = a1 * b2 - a2 * b1
        if D == 0:
            return None
        x = (b1 * c2 - b2 * c1) / D
        y = (a2 * c1 - a1 * c2) / D
        return int(x), int(y)



    def get_max_contours_lines(self,img,gray):
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # thresh = cv2.adaptiveThreshold(gray, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=int(1080 / 3 - 1), C=20)
        # thresh = cv2.adaptiveThreshold(gray, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
        #                                thresholdType=cv2.THRESH_BINARY_INV, blockSize=9, C=5)

        cv2.imshow("thresh", thresh)
        # 进行形态学操作，去除噪声和小区域
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算
        # cv2.imshow("opening", opening)
        closeing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算
        # 进行连通区域分析
        # cv2.imshow("closeing", closeing)

        new_closing = self.reverse_black_white(closeing)
        cv2.imshow("new_closing", new_closing)

        # canny边缘
        # canny_edge = cv2.Canny(new_closing, 128, 255)
        #
        # cv2.imshow("canny", canny_edge)

        # 找出轮廓
        contours, hierarchy = cv2.findContours(new_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(canny_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)
        # 找出最大面积，然后在最大面积基础上做直线
        if contours:
            cv2.drawContours(img, [contours[0]], -1, (250, 0, 0), 3)
            cv2.imshow("drawcontours", img)
        mask = np.zeros(new_closing.shape, dtype="uint8")
        cv2.drawContours(mask, [contours[0]], -1, (250, 255, 255), 1)
        cv2.imshow("mask_contours",mask)
        lines = cv2.HoughLinesP(mask, 1, np.pi / 3600, 140, minLineLength=180, maxLineGap=80)
        for line in lines:
            # print("lines = ", line)
            # 获取坐标
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)
        cv2.imshow("imagq", img)


        return lines
        # 获取直线函数

        # 找出直线
        # canny_edge = cv2.Canny(mask, 128, 255)
        # cv2.imshow("canny", canny_edge)
        # for cnt in contours:
        #     # if len(cnt)>4:
        #     print("cnt = ",cnt)
        #     # rect = cv2.minAreaRect(cnt)
        #     # box = cv2.boxPoints(rect)
        #     # box = np.int0(box)q
        #     # cv2.drawContours(img,[box],0,(0,0,255),2)
        #     epsilon = 0.00001 * cv2.arcLength(cnt, True)
        #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        #     cv2.drawContours(img, approx, -1, (0, 255, 0), 5)
        #     cv2.polylines(img, [approx], True, (0, 0, 255), 2)




        # lines = cv2.HoughLinesP(canny_edge, 1, np.pi / 180, 100, minLineLength=230, maxLineGap=80)

        # 笔在哪一边时候，那边的边不确定



        # n, labels, stats, centroids = cv2.connectedComponentsWithStats(new_closing)
        # area_label_list = list(np.where((stats[:, 4] < 1000000) & (stats[:, 4] > 100000))[0])
        # # 找到最大连通区域
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
        #
        # # 提取最大连通区域
        # mask = np.zeros_like(gray)
        # for area_label in area_label_list:
        #     mask[labels == area_label] = 255
        # cv2.imshow("mask_connect", mask)
        # # 进行形态学操作，填充区域
        # kernel_5x5 = np.ones((5, 5), np.uint8)
        # kernel_3x3 = np.ones((3, 3), np.uint8)
        # # mask_er = cv2.dilate(mask, kernel_5x5, iterations=50)
        # mask_er = mask
        # closed_mask = cv2.morphologyEx(mask_er, cv2.MORPH_CLOSE, kernel_5x5)

        # # 进行轮廓检测
        # contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.imshow("close_mask", closed_mask)
    @staticmethod
    def screen_mask(img,screen_rect):
        if screen_rect:
            mask = np.zeros(img.shape[:2], dtype="uint8")
            # print(screen_rect)
            cv2.rectangle(mask, (screen_rect[0], screen_rect[1]), (screen_rect[2], screen_rect[3]), 255, -1)

            screen_masked = cv2.bitwise_and(img, img, mask=mask)

            # cv2.imshow("rectangular mask", masked)
        else:
            screen_masked = copy.deepcopy(img)
            # crop_image = img
        return screen_masked
    @staticmethod
    def judge_location_pen(pen_point,screen_rect):
        buffer = 150 # 左右角落缓冲区
        location = "out"
        if screen_rect:
            # 笔不在电视板内
            if (pen_point[0] < (screen_rect[0]-50) or (pen_point[0] > (screen_rect[2] + 50))) or\
                    ((pen_point[1] < (screen_rect[1] - 50)) or (pen_point[1] > screen_rect[3]+50)):
                location = "out"
            else:
                if abs(pen_point[0]-screen_rect[0])<abs(pen_point[0]-screen_rect[2]):
                    # 在左边
                    if abs(pen_point[0]-screen_rect[0])<abs(pen_point[1]-screen_rect[1]):
                        # 离左边更近 在左边
                        if abs(pen_point[1]-screen_rect[1])<buffer:
                            location = "left_buffer"
                        else:
                            location = "left"

                    else:
                        # 在上面
                        location = "up"
                        pass
                    pass
                elif abs(pen_point[0]-screen_rect[0])>abs(pen_point[0]-screen_rect[2]):
                    # 在右边
                    if abs(pen_point[0]-screen_rect[0])<abs(pen_point[0]-screen_rect[3]):
                        # 在右边
                        if abs(pen_point[0] - screen_rect[3])<buffer:
                            location = "right_buffer"
                        else:
                            location = "right"

                    else:
                        # 在上边
                        location = "up"

        return location

    @staticmethod
    def judge_location_center(pen_point):
        x = pen_point[0]
        y = pen_point[1]
        location = "上"
        if x < 960:
            if y <540:
                location = "左"
            else:
                location = "右"


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

    # @staticmethod
    def egde_detect_biology(self,img,gray,pen_point):
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

        cv2.imshow("thresh", thresh)
        # 进行形态学操作，去除噪声和小区域
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算
        # cv2.imshow("opening", opening)
        closeing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭运算
        # 进行连通区域分析
        # cv2.imshow("closeing", closeing)

        new_closing = self.reverse_black_white(closeing)
        cv2.imshow("new_closing", new_closing)

        # canny边缘
        canny_edge = cv2.Canny(new_closing,128,255)

        cv2.imshow("canny",canny_edge)



        # 找出轮廓
        contours, hierarchy = cv2.findContours(new_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierarchy = cv2.findContours(canny_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        contours = sorted(contours,key=lambda c: cv2.contourArea(c), reverse=True)

            # 找出最大面积，然后在最大面积基础上做直线
        if contours:
            cv2.drawContours(img, [contours[0]], -1, (250, 0, 0), 3)
            cv2.imshow("drawcontours", img)


        # for cnt in contours:
        #     # if len(cnt)>4:
        #     print("cnt = ",cnt)
        #     # rect = cv2.minAreaRect(cnt)
        #     # box = cv2.boxPoints(rect)
        #     # box = np.int0(box)q
        #     # cv2.drawContours(img,[box],0,(0,0,255),2)
        #     epsilon = 0.00001 * cv2.arcLength(cnt, True)
        #     approx = cv2.approxPolyDP(cnt, epsilon, True)
        #     cv2.drawContours(img, approx, -1, (0, 255, 0), 5)
        #     cv2.polylines(img, [approx], True, (0, 0, 255), 2)

        lines = cv2.HoughLinesP(canny_edge, 1, np.pi / 180, 100, minLineLength=230, maxLineGap=80)

        # 笔在哪一边时候，那边的边不确定
        # up = []
        # down = []
        # left = []
        # right = []
        # coords = []
        # for i,line in enumerate(lines):
        #     x1, y1, x2, y2 = line[0]
        #     if abs(y2-y1)>abs(x2-x1):
        #         # y轴方向 y轴变化大
        #         if x1<960 and x2<960:
        #             coord = "left"
        #             left.append(line[0])
        #         elif x1>960 and x2>960:
        #             coord = "right"
        #             right.append(line[0])
        #         else:
        #             coord = ""
        #     else:
        #         # x轴方向 x轴变化大
        #         if y1<540 and y2<540:
        #             coord = "up"
        #             up.append(line[0])
        #         elif y1>540 and y2>540:
        #             coord = "down"
        #             down.append(line[0])
        #         else:
        #             coord = ""
        #     coords.append(coord)


        # print(coords)
        for line in lines:
            print("lines = ",line)
            # 获取坐标
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), thickness=5)
        cv2.imshow("imagq", img)





        n, labels, stats, centroids = cv2.connectedComponentsWithStats(opening)
        area_label_list = list(np.where((stats[:, 4] < 1000000) & (stats[:, 4] > 100000))[0])
        #找到最大连通区域
        max_area = 0
        max_label = 0
        for i in range(0, n):
            if stats[i, cv2.CC_STAT_AREA] > max_area:
                max_area = stats[i, cv2.CC_STAT_AREA]
                max_label = i
        stats[np.argsort(-stats[:, 4])][1:3, :]
        # 提取最大连通区域
        mask = np.zeros_like(gray)
        mask[labels == max_label] = 255

        # 提取最大连通区域
        mask = np.zeros_like(gray)
        for area_label in area_label_list:
            mask[labels == area_label] = 255
        cv2.imshow("mask_connect", mask)
        # 进行形态学操作，填充区域
        kernel_5x5 = np.ones((5, 5), np.uint8)
        kernel_3x3 = np.ones((3, 3), np.uint8)
        # mask_er = cv2.dilate(mask, kernel_5x5, iterations=50)
        mask_er=mask
        closed_mask = cv2.morphologyEx(mask_er, cv2.MORPH_CLOSE, kernel_5x5)

        # # 进行轮廓检测
        # contours, hierarchy = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("close_mask",closed_mask)
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
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_mask, connectivity=4)
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
        # contours, hierarchy = cv2.findContours(gray_mask_ode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(gray_mask_ode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        screen_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            # if w ==1920 and h==1080:
            #     continue
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
    # video = cv2.VideoCapture("./data/0314_left.mp4")
    # video = cv2.VideoCapture("./data/0302_4mm.mp4")
    video = cv2.VideoCapture("./data/20230308_4mm.mp4")

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


        image = cv2.resize(image,(1920,1080))
        cv2.imshow("image", image)
        print(image.shape)
        frame = rectify_video(image, map1, map2)
        new_frame = copy.deepcopy(frame)
        # frame=image

        # frame = cv2.resize(frame, (960, 540))
        # cv2.imshow('frame', frame)

        # 边框检测

        # 算法模型
        results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 没有好的框 可以左提醒
        detect_result = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
        for box in detect_result:
            l, t, r, b = int(box["xmin"]),int(box["ymin"]),int(box["xmax"]),int(box["ymax"])
            confidence = round(box["confidence"],1)
            cls_name = box["name"]
            # outside = t - h >= 3
            if cls_name=="pen" or cls_name=="screen":
                cv2.rectangle(new_frame, (l, t), (r, b), (0, 255, 0), 2)
                cv2.putText(new_frame, cls_name + "-" + str(confidence), (l, t), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        cv2.imshow("ff",new_frame)
        # results.show()

        # 当前笔的坐标,笔的位置，屏幕位置
        pen_point,pen_center,pen_rect,screen_rect = current_coord(detect_result,results)

        # 边框提取
        edgeDetect.edge_handle(frame,screen_rect,pen_point)
        # A = abase()
        # # A (xx,yy,zz)

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