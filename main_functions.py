# -*- coding: utf-8 -*-

import numpy as np
import cv2
from small_support_functions import *
from display_functions import *
import config
        

def extract_features(image_list, use_ratio, resize_ratio, resized_image_list, image_name_list, draw_mask):
    """
    进行SIFT特征点提取

    :param image_list: 图片列表
    :param use_ratio: 图片尺度缩放比
    :param resize_ratio: 显示图片放缩比
    :param resized_image_list: 显示图片列表
    :param image_name_list: 图片名称列表
    :return: 特征点列表，用于显示的特征点列表，特征向量列表，色彩列表
    """
    total_size = len(image_list)

    sift_model = cv2.xfeatures2d.SIFT_create()
    kp_list = []
    resized_kp_list = []
    desc_list = []
    colors_list = []
    for i in range(total_size):
        org_image = image_list[i]
        x, y = org_image.shape[0:2]
        use_image = cv2.resize(org_image, (int(y / use_ratio), int(x / use_ratio)))
        # 掩模绘制
        if draw_mask:
            mask = draw_mask(resized_image_list[i], resize_ratio,
                             image_name_list[i].strip(".jpg").strip(".jpeg") + "_mask.txt")
            use_kp, desc = sift_model.detectAndCompute(cv2.cvtColor(use_image, cv2.COLOR_RGB2GRAY), mask)

        else:
            use_kp, desc = sift_model.detectAndCompute(cv2.cvtColor(use_image, cv2.COLOR_RGB2GRAY), None)
        kp = []
        r_kp = []
        colors = []
        for use_kp_item in use_kp:
            x = use_kp_item.pt[0] * use_ratio
            y = use_kp_item.pt[1] * use_ratio
            color = image_list[i][int(y)][int(x)]
            colors.append(color)
            r_kp_item = cv2.KeyPoint()
            r_kp_item.pt = (x / resize_ratio, y / resize_ratio)
            r_kp.append(r_kp_item)
            kp_item = cv2.KeyPoint()
            kp_item.pt = (x, y)
            kp.append(kp_item)
        kp_list.append(kp)
        resized_kp_list.append(r_kp)
        colors_list.append(colors)
        desc_list.append(desc)
        print("features extraction complete:" + str(i + 1) + " kp_cnts:" + str(len(kp)))
        if config.show_features:
            key = show_keypoints(resized_image_list[i], r_kp, "keypoints_" + str(i), True)
            if key == 27:
                config.show_features = False
    return kp_list, resized_kp_list, desc_list, colors_list


def match_features(desc_list, rimgs=None, rkps=None, show_matches=False):
    """
    特征点匹配

    :param desc_list: 特征向量列表
    :return: 初筛选得到的特征点对
    """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good_matches_list = []
    for i in range(len(desc_list) - 1):
        descs1 = desc_list[i]
        descs2 = desc_list[i + 1]
        matches = bf.knnMatch(descs1, descs2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < config.dist_coef * n.distance:
                good_matches.append([m])
        good_matches_list.append(good_matches)
        print("features matching complete:" + str(i + 1) + " match_cnts:" + str(len(good_matches)))
        if config.show_matches:
            key = show_match_results(rimgs[i], rkps[i], rimgs[i + 1], rkps[i + 1], good_matches, "matches_" + str(i), True)
            if key == 27:
                config.show_matches = False
    return good_matches_list


def init_structure(kps_list, colors_list, good_matches_list, K):
    """
    三维点云模型初始化

    :param kps_list: 特征点列表
    :param colors_list: 色彩列表
    :param good_matches_list: 初筛选特征点对列表
    :param K: 相机内参K
    :return: 三维点云, 旋转矩阵列表, 位移向量列表, 2D到3D点对应列表, 三维点色彩
    """
    kps1 = kps_list[0]
    kps2 = kps_list[1]
    colors1 = colors_list[0]
    matches = good_matches_list[0]
    pts1 = np.asarray([kps1[match[0].queryIdx].pt for match in matches])
    pts2 = np.asarray([kps2[match[0].trainIdx].pt for match in matches])

    # 本质矩阵求解
    ess_mat, inlier_points = cv2.findEssentialMat(pts1, pts2, K, cv2.FM_RANSAC, 0.999, 1.0)

    # 恢复相机外参
    _, R, t, tried_points = cv2.recoverPose(ess_mat, pts1, pts2, K, mask=inlier_points)

    tried_matches = []
    for num, match in enumerate(matches):
        if tried_points[num][0] == 1:
            tried_matches.append(match)

    tried_pts1 = np.asarray([kps1[match[0].queryIdx].pt for match in tried_matches])
    tried_pts2 = np.asarray([kps2[match[0].trainIdx].pt for match in tried_matches])
    tried_colors1 = np.asarray([colors1[match[0].queryIdx] for match in tried_matches])
    colors = tried_colors1
    R0 = np.eye(3, 3)
    t0 = np.zeros((3, 1))

    # 三角法还原空间点
    structure = triangulate(K, R0, t0, R, t, tried_pts1, tried_pts2)

    d2to3ds_list = [np.ones(len(kps), dtype=int) * -1 for kps in kps_list]
    idx = 0
    for match in tried_matches:
        d2to3ds_list[0][match[0].queryIdx] = idx
        d2to3ds_list[1][match[0].trainIdx] = idx
        idx += 1

    Rots = [R0, np.array(R)]
    trans = [t0, t]
    return structure, Rots, trans, d2to3ds_list, colors


def triangulate(K, R1, t1, R2, t2, pts1, pts2):
    """
    三角形法还原空间点

    :param K: 相机内参K
    :param R1: 旋转矩阵1
    :param t1: 位移向量1
    :param R2: 旋转矩阵2
    :param t2: 位移向量2
    :param pts1: 第一组像素点坐标
    :param pts2: 第二组像素点坐标
    :return: 对应的三维空间点坐标
    """
    cam1 = np.matmul(K, Rt2Cam_mat(R1, t1))
    cam2 = np.matmul(K, Rt2Cam_mat(R2, t2))
    kp4ds = cv2.triangulatePoints(cam1, cam2, pts1.T, pts2.T)
    kp3ds = []
    for j in range(len(kp4ds[0])):
        col = kp4ds[:, j]
        col /= col[3]
        kp3ds.append([col[0], col[1], col[2]])
    return np.array(kp3ds)


def prep_pnp_points(matches, d2to3ds, structure, kps):
    """
    准备PNP点对及空间点像素点坐标

    :param matches: 特征点对
    :param d2to3ds: 2D点到3D点对应关系
    :param structure: 已有三维点云
    :param kps: 新加入的一帧像素点
    :return: PNP用到的空间点，对应的像素点坐标，PNP点对
    """
    world_pts = []
    image_pts = []
    pnp_matches = []
    for match in matches:
        struct_idx = d2to3ds[match[0].queryIdx]
        if struct_idx >= 0:
            world_pts.append(structure[struct_idx])
            image_pts.append(kps[match[0].trainIdx].pt)
            pnp_matches.append(match)
    world_pts = np.array(world_pts)
    image_pts = np.array(image_pts)
    while len(image_pts) < 7:
        world_pts = np.append(world_pts, [world_pts[0]], axis=0)
        image_pts = np.append(image_pts, [image_pts[0]], axis=0)
    return world_pts, image_pts, pnp_matches


def prep_next_pts(K, R1, t1, R2, t2, kps1, kps2, matches, mass_center, colors1, err_purge_thresh, err_purge_percentage, dist_purge_thresh):
    """
    筛选将插入的空间点

    :param K: 相机内参K
    :param R1: 旋转矩阵1
    :param t1: 位移向量1
    :param R2: 旋转矩阵2
    :param t2: 位移向量2
    :param kps1: 第一组特征点
    :param kps2: 第二组特征点
    :param matches: 特征点配对
    :param mass_center: 点云质心
    :param err_purge_thresh: 重投影误差筛选阈值
    :param err_purge_percentage: 误差最大百分比筛选阈值
    :param dist_purge_thresh: 距质心距离筛选阈值
    :return: 将插入的空间点坐标，对应的特征点对
    """
    output_container = []
    kp3ds = []
    purged_matches = []
    new_colors = []

    pts1 = np.asarray([kps1[match[0].queryIdx].pt for match in matches])
    pts2 = np.asarray([kps2[match[0].trainIdx].pt for match in matches])
    ess_mat, inlier_points = cv2.findEssentialMat(pts1, pts2, K, cv2.FM_RANSAC, 0.999, 1.0)
    _, _, _, next_points = cv2.recoverPose(ess_mat, pts1, pts2, K, mask=inlier_points)
    for num, match in enumerate(matches):
        if next_points[num][0] == 1:
            pt1 = np.array([kps1[match[0].queryIdx].pt])
            pt2 = np.array([kps2[match[0].trainIdx].pt])
            kp3d = triangulate(K, R1, t1, R2, t2, pt1, pt2)
            err1 = reprojection_error(kp3d, R1, t1, K, pt1).ravel()
            err2 = reprojection_error(kp3d, R2, t2, K, pt2).ravel()
            this_max_err = max([abs(err1[0]), abs(err1[1]), abs(err2[0]), abs(err2[1])])
            if this_max_err > err_purge_thresh:
                continue
            dist = abs(kp3d[0] - mass_center)
            if dist[dist.argmax()] > dist_purge_thresh:
                continue
            else:
                output_container.append([kp3d[0], match, colors1[match[0].queryIdx], this_max_err])
    output_container.sort(key=lambda a: a[3])
    for i in range(int(len(output_container) * (1 - err_purge_percentage))):
        kp3ds.append(output_container[i][0])
        purged_matches.append(output_container[i][1])
        new_colors.append(output_container[i][2])

    return np.array(kp3ds).squeeze(), purged_matches, np.array(new_colors)


def fusion_structure(matches, d2to3ds, next_d2to3ds, structure, next_structure, struct_colors, colors1, R1, t1, R2, t2, K, kps1, kps2, purge_thresh):
    """
    空间点插入

    :param matches: 新插入的特征点对
    :param d2to3ds: 2D点到3D点对应关系
    :param next_d2to3ds: 下一个2D到3D点对应关系容器
    :param structure: 已有三维点云
    :param next_structure: 新插入的空间点
    :param struct_colors: 已有空间点色彩
    :param colors1: 新空间点色彩列表
    :param R1: 旋转矩阵1
    :param t1: 位移向量1
    :param R2: 旋转矩阵2
    :param t2: 位移向量2
    :param K: 内参矩阵
    :param kps1: 第一组特征点
    :param kps2: 第二组特征点
    :param purge_thresh: 重投影误差筛选阈值
    :return: 2D点到3D点对应关系，下一个2D到3D点对应关系容器，三维点云，三维点云色彩
    """
    for i, match in enumerate(matches):
        struct_idx = d2to3ds[match[0].queryIdx]
        if struct_idx >= 0:
            pt1 = np.array([kps1[match[0].queryIdx].pt])
            pt2 = np.array([kps2[match[0].trainIdx].pt])
            err1 = reprojection_error(structure[struct_idx], R1, t1, K, pt1).ravel()
            err2 = reprojection_error(structure[struct_idx], R2, t2, K, pt2).ravel()
            if abs(err1[0]) > purge_thresh or abs(err1[1]) > purge_thresh or abs(err2[0]) > purge_thresh or abs(err2[1]) > purge_thresh:
                continue
            else:
                next_d2to3ds[match[0].trainIdx] = struct_idx
        else:
            d2to3ds[match[0].queryIdx] = len(structure)
            next_d2to3ds[match[0].trainIdx] = len(structure)
            structure = np.append(structure, [next_structure[i]], axis=0)
            new_color = colors1[match[0].queryIdx]
            struct_colors = np.append(struct_colors, [new_color], axis=0)
    return d2to3ds, next_d2to3ds, structure, struct_colors


def reprojection_error(kp3ds, R, t, K, kp2ds):
    """
    重投影误差求取

    :param kp3ds: 空间点列表
    :param R: 旋转矩阵
    :param t: 位移向量
    :param K: 相机内参
    :param kp2ds: 对应的像素点列表
    :return: 重投影误差列表
    """
    proj, jac = cv2.projectPoints(kp3ds, R, t, K, np.array([]))
    return proj - kp2ds



