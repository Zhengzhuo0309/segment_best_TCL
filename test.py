import numpy as np
import cv2
# 轨迹直线拟合
def fit_contours_vertical(screen_contours_best_left_xy):  # 输入一边纵向轨迹，拟合直线; 格式 np.array([[x1,y1]，[x2,y2]...])
    block_left_y_max = screen_contours_best_left_xy[:, 1].max()
    block_left_y_min = screen_contours_best_left_xy[:, 1].min()
    # rows, cols = img_copy.shape[:2]
    [vx, vy, x_point, y_point] = cv2.fitLine(screen_contours_best_left_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    k_fit = vy / vx
    b_fit = y_point - k_fit * x_point
    # lefty = int((-x_point * vy / vx) + y_point)
    # righty = int(((cols - x_point) * vy / vx) + y_point)
    if np.count_nonzero(k_fit)!=0:
        left_x_max = int((block_left_y_max - b_fit) / k_fit)
        left_x_min = int((block_left_y_min - b_fit) / k_fit)
    else:                                      #k=0
        left_x_max = screen_contours_best_left_xy[:, 0].max()
        left_x_min = screen_contours_best_left_xy[:, 0].min()
    track_length = np.linalg.norm(np.array([left_x_min, block_left_y_min]) - np.array([left_x_max, block_left_y_max]))

    return k_fit, b_fit, track_length, (left_x_min, block_left_y_min), (left_x_max, block_left_y_max)

def fit_contours_horizontal(screen_contours_best_up_xy):  # 输入一边横向轨迹，拟合直线; 格式 np.array([[x1,y1]，[x2,y2]...])
    block_up_x_max = screen_contours_best_up_xy[:, 0].max()
    block_up_x_min = screen_contours_best_up_xy[:, 0].min()
    # rows, cols = img_copy.shape[:2]
    [vx, vy, x_point, y_point] = cv2.fitLine(screen_contours_best_up_xy, cv2.DIST_L2, 0, 0.01, 0.01)
    k_fit = vy / vx
    b_fit = y_point - k_fit * x_point
    # lefty = int((-x_point * vy / vx) + y_point)
    # righty = int(((cols - x_point) * vy / vx) + y_point)
    up_y_max = int(k_fit*block_up_x_max+b_fit)
    up_y_min = int(k_fit*block_up_x_min+b_fit)
    track_length = np.linalg.norm(np.array([block_up_x_min,up_y_min]) - np.array([block_up_x_max,up_y_max]))
    return k_fit, b_fit, track_length, (block_up_x_min,up_y_min), (block_up_x_max,up_y_max)


class Threshold(object):  # 阈值文件
    def __init__(self):
        self.fit_change_th = 15  # 拟合变化计数阈值
        self.in_direction_count_th = 5  # 处于该方向的次数阈值
        self.out_direction_count_th = 10  # 不在该方向上次数阈值
        self.out_direction_change_th = 20  # 不在该方向上变化计数阈值
        self.out_count_th = 10  # 区域外次数阈值


class Flag(object):
    def __init__(self):
        self.update_flag = False  # 长度匹配开关，分出一个轨迹段之后打开，匹配一次后关闭
        self.area_cut_flag = False  # 出重叠区域后分段
        self.area_in_flag = True  # 进入重叠区域
        self.init_notin_flag = False  # 初始是否处于重叠区域，默认在
        self.out_direction_dectect_flag = False  # 其他方向计数开关

    def reset_flag(self):
        self.update_flag = False  # 长度匹配开关，分出一个轨迹段之后打开，匹配一次后关闭
        self.out_direction_dectect_flag = False  # 其他方向计数开关


class Temp(object):
    def __init__(self):
        self.track_cur = []  # 当前用于拟合的轨迹
        self.track_cur_img = []  # 当前用于拟合的轨迹对应图片
        self.move_cur = []  # 被移除的轨迹点——下一段轨迹的开头
        self.move_cur_img = []  # 被移除的轨迹点对应的图片
        self.track_all = []  # 所有轨迹
        self.track_img_all = []  # 所有轨迹对应的图片
        self.track_direction_all = []  # 所有轨迹对应的方向
        self.track_with_two_position = []  # 两个区域重合的点
        self.track_with_two_position_img = []  # 两个区域重合的点对应图片

    def reset_tmp(self):
        self.track_cur = []  # 当前用于拟合的轨迹
        self.track_cur_img = []  # 当前用于拟合的轨迹对应图片
        self.move_cur = []  # 被移除的轨迹点——下一段轨迹的开头
        self.move_cur_img = []  # 被移除的轨迹点对应的图片——下一段轨迹的开头
        self.track_with_two_position = []  # 两个区域重合的点
        self.track_with_two_position_img = []  # 两个区域重合的点对应图片


class Context(object):

    def __init__(self):
        self.position = ''  # 当前位置
        self.direction = ''  # 当前轨迹方向
        self.change_direction = ''  # 轨迹点变化方向
        self.fit_change_num_x_plus = 0  # 拟合点x正方向变化计数
        self.fit_change_num_y_plus = 0  # 拟合点y正方向变化计数
        self.fit_change_num_x_minus = 0  # 拟合点x负方向变化计数
        self.fit_change_num_y_minus = 0  # 拟合点y负方向变化计数
        self.in_direction_count = 0  # 处于该方向的次数
        self.out_direction_count = 0  # 该方向确定之后，不在该方向的次数
        self.out_count = 0  # 跑出有效区域个数
        self.cur_area_count = 0  # 区域重合计数
        self.per_area_count = 0  # 上一帧区域重合计数
        self.position_dict = {'left': False, 'right': False, 'up': False}  # 位置重合区域字典
        self.per_position_dict = {'left': False, 'right': False, 'up': False}  # 前一时刻位置重合区域字典
        self.match_dict = {'left':  # 边框匹配字典
            {
                'y_plus': {'func': fit_contours_vertical, 'end_point': None, 'need_length': None},
                'y_minus': {'func': fit_contours_vertical, 'end_point': None, 'need_length': None},
                'k_fit': None,
                'b_fit': None
            },
            'right':
                {
                    'y_plus': {'func': fit_contours_vertical, 'end_point': None, 'need_length': None},
                    'y_minus': {'func': fit_contours_vertical, 'end_point': None, 'need_length': None},
                    'k_fit': None,
                    'b_fit': None
                },
            'up':
                {
                    'x_plus': {'func': fit_contours_horizontal, 'end_point': None, 'need_length': None},
                    'x_minus': {'func': fit_contours_horizontal, 'end_point': None, 'need_length': None},
                    'k_fit': None,
                    'b_fit': None
                },
        }
        self.track_match_dict = {  # 决策匹配字典
            'left':
                {
                    'y_plus': {'end_point': None, 'match_length': None},
                    'y_minus': {'end_point': None, 'match_length': None},
                },
            'right':
                {
                    'y_plus': {'end_point': None, 'match_length': None},
                    'y_minus': {'end_point': None, 'match_length': None},
                },
            'up':
                {
                    'x_plus': {'end_point': None, 'match_length': None},
                    'x_minus': {'end_point': None, 'match_length': None},
                },
        }
        self.done_dict = {'left': 0, 'right': 0, 'up': 0}  # 结果字典

    def reset_context(self):
        self.position = ''  # 当前位置
        self.direction = ''  # 当前轨迹方向
        self.change_direction = ''  # 轨迹点变化方向

        self.in_direction_count = 0  # 处于该方向的次数
        self.out_direction_count = 0  # 该方向确定之后，不在该方向的次数
        self.out_count = 0  # 跑出有效区域个数


class Config(Threshold, Flag, Temp, Context):  # 配置文件
    def __init__(self):
        super(Threshold,self).__init__()
        super(Flag,self).__init__()
        super(Temp,self).__init__()
        super(Context,self).__init__()

    def reset_config(self):  # 重置所有计数器及容器 —— 拟合一次后调用
        self.reset_tmp()
        self.reset_flag()
        self.reset_context()
config = Config()