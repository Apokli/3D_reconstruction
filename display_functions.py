# -*- coding: utf-8 -*-

import cv2
import numpy as np
from math import pow, sqrt
from copy import deepcopy
from small_support_functions import *
from config import *
from mayavi import mlab
from matplotlib import pyplot as plt
import matplotlib


def show_keypoints(rimg, rkps, name="keypoints", console=True):
    results = cv2.drawKeypoints(rimg, rkps, rimg, color=(255, 0, 255))
    if console:
        cv2.imshow(name, results)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return results


def show_match_results(rimg1, rkps1, rimg2, rkps2, matches, name="match results", console=True):
    """
    function for displaying keypoint matches in a resized image.

    :param rimg1: first resized image
    :param rkps1: first set of resized keypoints
    :param rimg2: second resized image
    :param rkps2: second set of resized keypoints
    :param matches: keypoint matches
    :param name: name of the OpenCV display window
    :return:
    """
    results = cv2.drawMatchesKnn(rimg1, rkps1, rimg2, rkps2, matches, None, flags=2)
    if console:
        cv2.imshow(name, results)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        return results

# ----------------------------------------------- 手动剔除误匹配点 -------------------------------------------

global base_img
global show_img

def manual_purge(fund_mat, match_points, kps1, kps2, rkps1, rkps2, rimg1, rimg2):
    """

    :param fund_mat: Fundamental matrix
    :param match_points: keypoint matches
    :param kps1: first list of keypoints
    :param kps2: second list of keypoints
    :param rkps1: first list of keypoints
    :param rkps2:
    :param rimg1:
    :param rimg2:
    :return:
    """
    global base_img
    global show_img
    score_list = []
    clean_matches = []
    for match in match_points:
        clean_matches.append(match)
        score = get_match_score(fund_mat, kps1, kps2, match)
        score_list.append(score)

    purge_complete = False
    deleted_matches = []
    base_img = cv2.drawMatchesKnn(rimg1, rkps1, rimg2, rkps2, match_points, None, flags=2)
    show_img = deepcopy(base_img)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse_action_manual_purge, param=(rkps1, rkps2, clean_matches,
                                                                       rimg1, rimg2, deleted_matches, score_list))

    while not purge_complete:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(show_img, "left click on point to purge, right click to undo", (50, 50), font, 1, (255, 0, 0), 3)
        cv2.imshow("image", show_img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return match_points
        elif key == 32:
            cv2.destroyAllWindows()
            return clean_matches


def on_mouse_action_manual_purge(event, x, y, flags, param):
    global base_img
    global show_img
    rkps1 = param[0]
    rkps2 = param[1]
    matches = param[2]
    rimg1 = param[3]
    rimg2 = param[4]
    del_matches = param[5]
    scores = param[6]
    x_gap = param[3].shape[1]

    show_img = deepcopy(base_img)

    p1 = None
    p2 = None
    score = None
    selected = False

    for num, match in enumerate(matches):
        p1 = rkps1[match[0].queryIdx]
        p2 = rkps2[match[0].trainIdx]
        score = scores[num]
        if pow((x - p1.pt[0]), 2) + pow((y - p1.pt[1]), 2) < 12 or \
                pow((x - x_gap - p2.pt[0]), 2) + pow((y - p2.pt[1]), 2) < 12:
            selected = True
            break

    if event == cv2.EVENT_MOUSEMOVE and selected:
        highlight_p1 = (int(p1.pt[0]), int(p1.pt[1]))
        highlight_p2 = (int(p2.pt[0]) + x_gap, int(p2.pt[1]))
        cv2.circle(show_img, highlight_p1, 5, (0, 0, 255), thickness=-1)
        cv2.circle(show_img, highlight_p2, 5, (0, 0, 255), thickness=-1)
        cv2.line(show_img, highlight_p1, highlight_p2, (0, 0, 255), 3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(show_img, "Dist:" + str(score), (50, 100), font, 1, (255, 0, 0), 3)

    elif event == cv2.EVENT_LBUTTONDOWN and selected:
        del matches[num]
        del_matches.append(match)
        base_img = cv2.drawMatchesKnn(rimg1, rkps1, rimg2, rkps2, matches, None, flags=2)

    elif event == cv2.EVENT_RBUTTONDOWN:
        matches.append(del_matches[-1])
        del del_matches[-1]
        base_img = cv2.drawMatchesKnn(rimg1, rkps1, rimg2, rkps2, matches, None, flags=2)

# ----------------------------------------------- 掩模制作 ---------------------------------------------


def draw_mask(_resized_image, _resize_ratio, mask_name=""):
    """
    draw a mask on the image. left button to set anchor points, right button to undo a point.

    :param rimg: a resized image to draw a mask on
    :param resize_ratio: ratio of resize
    :param mask_name: the mask txt file name
    :return: the mask
    """
    point_set = [True]
    resized_point_set = [True]
    resized_image = _resized_image
    resize_ratio = _resize_ratio
    _mask = None
    original_image_shape = (resized_image.shape[0] * resize_ratio, resized_image.shape[1] * resize_ratio)
    if use_available_mask:
        file = open(mask_name, 'r')
        for line in file.readlines():
            str_nums = line.split()
            nums = [int(num) for num in str_nums]
            point_set.append(tuple(nums))
        if acquiesce_mask:
            point_set[0] = False
            resized_point_set[0] = False
            if len(point_set) == 1:
                return np.ones(original_image_shape, dtype=np.uint8)
            else:
                base = np.zeros(original_image_shape, dtype=np.uint8)
                polygon = np.array([point_set[1:]])
                _mask = cv2.fillPoly(base, polygon, 1)
                return _mask

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_button_down_draw_mask,  param=(point_set, resized_point_set, resize_ratio))

    while True:
        mask_drawn_image = visualize_mask(resized_image, resized_point_set, point_set)
        if not point_set[0]:
            base = np.zeros(original_image_shape, dtype=np.uint8)
            polygon = np.array([point_set[1:]])
            _mask = cv2.fillPoly(base, polygon, 1)
        cv2.imshow("image", mask_drawn_image)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            return None
        elif key == 32:
            cv2.destroyAllWindows()
            if _mask is None:
                _mask = np.ones(original_image_shape, dtype=np.uint8)
            if save_drawn_mask:
                file = open(mask_name, 'w')
                for point in point_set[1:]:
                    file.write(str(point[0]) + " " + str(point[1]) + "\n")
            return _mask


def on_button_down_draw_mask(event, x, y, flags, param):
    point_set = param[0]
    resized_point_set = param[1]
    drawable = param[0][0]
    resize_ratio = param[2]
    if event == cv2.EVENT_LBUTTONDOWN and drawable:
        point_set.append((x * resize_ratio, y * resize_ratio))
        resized_point_set.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(point_set) > 1:
            del point_set[-1]
            del resized_point_set[-1]


def visualize_mask(_resized_image, _resized_point_set, _point_set):
    resized_point_set = _resized_point_set
    resized_image = _resized_image
    point_set = _point_set
    img = deepcopy(resized_image)
    _mask = None
    resized_point_set[0] = True

    for num, point in enumerate(resized_point_set):
        if num == 0:
            pass
        else:
            if num == 1:
                cv2.circle(img, point, 6, (0, 200, 0), thickness=-1)
            else:
                cv2.circle(img, point, 4, (0, 0, 255), thickness=-1)
                cv2.line(img, point, resized_point_set[num - 1], (0, 0, 255), 3)
                if num == len(resized_point_set) - 1 and sqrt(pow(resized_point_set[1][0] - point[0], 2) +
                                                              pow(resized_point_set[1][1] - point[1], 2)) < 8:
                    resized_point_set[-1] = resized_point_set[1]
                    point_set[-1] = point_set[1]
                    resized_point_set[0] = False
                    point_set[0] = False
                    mask_thresh = 50
                    base = np.zeros(img.shape[:2], dtype=np.uint8)
                    polygon = np.array([resized_point_set[1:]])
                    _mask = cv2.fillPoly(base, polygon, 1)
                    bbox_mask = (255 * _mask > mask_thresh).astype(np.uint8)
                    color_mask = np.array([0, 215, 255], dtype=np.uint8)
                    bbox_mask = bbox_mask.astype(np.bool)
                    img[bbox_mask] = img[bbox_mask] * 0.5 + color_mask * 0.5
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, "Press Space to proceed", (50, 50), font, 1, (255, 0, 0), 3)
    return img


def show_3d_points(structure, struct_colors, name="3d_points"):
    """
    显示三维点云

    :param structure: 三维点云
    :param struct_colors: 三维点云色彩
    :param name: 窗口名
    :return:
    """
    display_colors = np.ones((len(struct_colors), 4), dtype=np.uint8)
    display_colors[:, 0] = struct_colors[:, 2]
    display_colors[:, 1] = struct_colors[:, 1]
    display_colors[:, 2] = struct_colors[:, 0]
    display_colors[:, 3] = 255
    pts = mlab.pipeline.scalar_scatter(structure[:, 0], structure[:, 1],
                                       structure[:, 2])  # plot the points
    pts.add_attribute(display_colors, 'colors')  # assign the colors to each point
    pts.data.point_data.set_active_scalars('colors')
    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = 0.1  # set scaling for all the points
    g.glyph.scale_mode = 'data_scaling_off'  # make all the points same size

    mlab.show()


def plot_residuals(residuals):
    """
    绘制重投影误差像素分布

    :param residuals: 重投影误差列表
    :return:
    """
    plt.figure()
    plt.plot(residuals)
    plt.show()


def plot_residual_distribution(residuals):
    """
    绘制重投影误差大小分布

    :param residuals: 重投影误差列表
    :return:
    """
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    label_list = ['0~2', '2~6', '6~10', '>10']  # 横坐标刻度显示值
    total_cnt = len(residuals)
    cnt02 = 0
    cnt26 = 0
    cnt610 = 0
    cnt10b = 0
    for res in residuals:
        if res > 2:
            if res > 6:
                if res > 10:
                    cnt10b += 1
                    continue
                cnt610 += 1
                continue
            cnt26 += 1
            continue
        cnt02 += 1
    num_list = [cnt02 / total_cnt, cnt26 / total_cnt, cnt610 / total_cnt, cnt10b / total_cnt]

    x = range(len(num_list))
    rects = plt.bar(x=x, height=num_list, width=0.4, alpha=0.8, color='blue')
    plt.ylim(0, 1)
    plt.ylabel("占比")
    plt.xticks(x, label_list)
    plt.xlabel("像素误差区间")
    plt.title("重投影误差值区间分布")

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 0.02, "{:.3}".format(height), ha="center", va="bottom")
    plt.show()