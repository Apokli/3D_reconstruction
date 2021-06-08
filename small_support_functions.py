# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os


def get_photos(route, display_size=(800, 1200)):
    """
    读取要重建的图片

    :param route: 文件路径
    :param display_size: 显示图片大小
    :return: 显示图片缩放比例，图片列表，图片名称列表，图片路径列表，显示图片列表
    """
    image_list = []
    image_complete_name_list = []
    image_name_list = []
    resized_image_list = []
    for root, dirs, files in os.walk(route):
        for file in files:
            if len(file) != len(file.strip(".jpg").strip(".jpeg")):
                image_complete_name = root + '/' + file
                cv_image = cv2.imread(image_complete_name)
                image_name_list.append(file)
                image_complete_name_list.append(image_complete_name)
                image_list.append(cv_image)
                x, y = cv_image.shape[0:2]
                resize_ratio = max(x // display_size[0], y // display_size[1])
                resized_image = cv2.resize(cv_image, (int(y / resize_ratio), int(x / resize_ratio)))
                resized_image_list.append(resized_image)
    return resize_ratio, np.array(image_list), np.array(image_name_list), np.array(image_complete_name_list), np.array(resized_image_list)


def get_K(route):
    """
    从txt读取相机内参

    :param route: 文件路径
    :return: 相机内参K
    """
    K = []
    try:
        Ktxt = open(route + "/K.txt", 'r', encoding="utf-8")
        for line in Ktxt.readlines():
            str_nums = line.strip("[]\n").replace(" ", "").split(',')
            K.append([float(str_num) for str_num in str_nums])
    except:
        return None
    return np.array(K)


def normalize_set(kps):
    """
    Function for normalizing keypoints

    :param kps: the keypoints set being normalized
    :return: (the normalized set, normalization matrix)
    """
    normalized_set = []
    size = len(kps)
    mean_x = 0
    mean_y = 0
    for kp in kps:
        mean_x += kp.pt[0]
        mean_y += kp.pt[1]
    mean_x /= size
    mean_y /= size

    mean_dev_x = 0
    mean_dev_y = 0
    for kp in kps:
        n_kp = cv2.KeyPoint()
        n_kp.pt = (kp.pt[0] - mean_x, kp.pt[1] - mean_y)
        normalized_set.append(n_kp)
        mean_dev_x += abs(n_kp.pt[0])
        mean_dev_y += abs(n_kp.pt[1])
    mean_dev_x /= size
    mean_dev_y /= size
    for n_kp in normalized_set:
        n_kp.pt = (n_kp.pt[0] / mean_dev_x, n_kp.pt[1] / mean_dev_y)
    H = np.array([[1 / mean_dev_x, 0, -mean_x / mean_dev_x],
                  [0, 1 / mean_dev_y, -mean_y / mean_dev_y],
                  [0, 0, 1]])
    return normalized_set, H


def get_match_score(F_mat, kps1, kps2, match):
    """

    :param F_mat: Fundamental matrix
    :param kps1: keypoints set 1
    :param kps2: keypoints set 2
    :param match: keypoint match
    :return: the distance of the match
    """
    kp1 = kps1[match[0].queryIdx].pt
    kp2 = kps2[match[0].trainIdx].pt
    p1 = np.array([kp1[0], kp1[1], 1])
    p2 = np.array([kp2[0], kp2[1], 1])
    score = np.matmul(np.matmul(p2, F_mat), p1.transpose())
    return score


def Rt2Cam_mat(R, t):
    """
    renders R and t into a complete camera matrix P

    :param R: Rotation matrix
    :param t: translation matrix
    :return: Cam_mat: the camera matrix P
    """
    Cam_mat = np.zeros((3, 4))
    Cam_mat[0:3, 0:3] = np.float32(R)
    Cam_mat[:, 3] = np.float32(t.T)

    return Cam_mat

