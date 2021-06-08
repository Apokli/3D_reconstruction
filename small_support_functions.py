# -*- coding: utf-8 -*-

import numpy as np
import cv2
import os
import config


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


def load_config(file_route):
    cnt = 0
    try:
        config_txt = open(file_route + "/config.txt", 'r', encoding="utf-8")
    except:
        print("无法打开或找不到config.txt")
        return False
    for line in config_txt.readlines():
        if line == "\n":
            continue
        title, value = line.strip().replace(" ", "").split('=')
        if title == "use_ratio":
            cnt += 1
            config.use_ratio = float(value)
        if title == "dist_coef":
            cnt += 1
            config.dist_coef = float(value)
        if title == "mass_dist":
            cnt += 1
            config.mass_dist = float(value)
        if title == "rep_err":
            cnt += 1
            config.rep_err = float(value)
        if title == "rep_per":
            cnt += 1
            config.rep_per = float(value)
        if title == "draw_mask":
            cnt += 1
            if value == "True":
                config.draw_mask = True
            elif value == "False":
                config.draw_mask = False
            else:
                cnt -= 1
        if title == "use_available_mask":
            cnt += 1
            if value == "True":
                config.use_available_mask = True
            elif value == "False":
                config.use_available_mask = False
            else:
                cnt -= 1
        if title == "acquiesce_mask":
            cnt += 1
            if value == "True":
                config.acquiesce_mask = True
            elif value == "False":
                config.acquiesce_mask = False
            else:
                cnt -= 1
        if title == "save_drawn_mask":
            cnt += 1
            if value == "True":
                config.save_drawn_mask = True
            elif value == "False":
                config.save_drawn_mask = False
            else:
                cnt -= 1
        if title == "show_features":
            cnt += 1
            if value == "True":
                config.show_features = True
            elif value == "False":
                config.show_features = False
            else:
                cnt -= 1
        if title == "show_matches":
            cnt += 1
            if value == "True":
                config.show_matches = True
            elif value == "False":
                config.show_matches = False
            else:
                cnt -= 1
        if title == "show_init_cloud":
            cnt += 1
            if value == "True":
                config.show_init_cloud = True
            elif value == "False":
                config.show_init_cloud = False
            else:
                cnt -= 1
        if title == "show_new_kp3ds":
            cnt += 1
            if value == "True":
                config.show_new_kp3ds = True
            elif value == "False":
                config.show_new_kp3ds = False
            else:
                cnt -= 1
        if title == "show_raw_structure":
            cnt += 1
            if value == "True":
                config.show_raw_structure = True
            elif value == "False":
                config.show_raw_structure = False
            else:
                cnt -= 1
        if title == "show_raw_residuals":
            cnt += 1
            if value == "True":
                config.show_raw_residuals = True
            elif value == "False":
                config.show_raw_residuals = False
            else:
                cnt -= 1
        if title == "show_tmp_structure":
            cnt += 1
            if value == "True":
                config.show_tmp_structure = True
            elif value == "False":
                config.show_tmp_structure = False
            else:
                cnt -= 1
        if title == "show_tmp_residuals":
            cnt += 1
            if value == "True":
                config.show_tmp_residuals = True
            elif value == "False":
                config.show_tmp_residuals = False
            else:
                cnt -= 1
        if title == "show_final_structure":
            cnt += 1
            if value == "True":
                config.show_final_structure = True
            elif value == "False":
                config.show_final_structure = False
            else:
                cnt -= 1
    if cnt == 18:
        print("config successfully loaded!")
        return True
    else:
        print("missing parameters or check spelling \"True\", \"False\"")
        return False


def save_config(file_route):
    config_txt = open(file_route + "/config.txt", 'w', encoding="utf-8")
    config_txt.write("use_ratio = " + str(config.use_ratio) + "\n")
    config_txt.write("dist_coef = " + str(config.dist_coef) + "\n")
    config_txt.write("\n")
    config_txt.write("mass_dist = " + str(config.mass_dist) + "\n")
    config_txt.write("rep_err = " + str(config.rep_err) + "\n")
    config_txt.write("rep_per = " + str(config.rep_per) + "\n")
    config_txt.write("\n")
    config_txt.write("draw_mask = " + str(config.draw_mask) + "\n")
    config_txt.write("use_available_mask = " + str(config.use_available_mask) + "\n")
    config_txt.write("acquiesce_mask = " + str(config.acquiesce_mask) + "\n")
    config_txt.write("save_drawn_mask = " + str(config.save_drawn_mask) + "\n")
    config_txt.write("\n")
    config_txt.write("show_features = " + str(config.show_features) + "\n")
    config_txt.write("show_matches = " + str(config.show_matches) + "\n")
    config_txt.write("show_init_cloud = " + str(config.show_init_cloud) + "\n")
    config_txt.write("show_new_kp3ds = " + str(config.show_new_kp3ds) + "\n")
    config_txt.write("show_raw_structure = " + str(config.show_raw_structure) + "\n")
    config_txt.write("show_raw_residuals = " + str(config.show_raw_residuals) + "\n")
    config_txt.write("show_tmp_structure = " + str(config.show_tmp_structure) + "\n")
    config_txt.write("show_tmp_residuals = " + str(config.show_tmp_residuals) + "\n")
    config_txt.write("show_final_structure = " + str(config.show_final_structure) + "\n")


