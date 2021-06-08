# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
from main_functions import *
from display_functions import *
from small_support_functions import *
from bundle_adjustment import PyBA
import config


if __name__ == '__main__':

    # 配置文件读入
    if have_config_txt:
        loaded = load_config(config.file_route)
        if not loaded:
            sys.exit()
    else:
        save_config(config.file_route)

    # 文件读入与图片处理
    resize_ratio, image_list, image_name_list, image_complete_name_list, resized_image_list = get_photos(config.file_route)

    # 读入内参矩阵K
    K_intrinsics = get_K(config.file_route)

    # 特征提取
    kps_list, resized_kps_list, desc_list, colors_list = extract_features(image_list, config.use_ratio, resize_ratio, resized_image_list, image_complete_name_list, config.draw_mask)

    # 特征匹配
    good_matches_list = match_features(desc_list, resized_image_list, resized_kps_list, config.show_matches)

    # 模型初始化
    structure, Rots, trans, d2to3ds_list, struct_colors = init_structure(kps_list, colors_list, good_matches_list, K_intrinsics)

    if config.show_init_cloud:
        show_3d_points(structure, struct_colors)

    # 增量式sfm
    for i in range(1, len(good_matches_list)):

        print("trial " + str(i) + ":")

        mass_center = np.mean(structure, axis=0)

        # 求解PNP问题
        world_pts, image_pts, pnp_matches = prep_pnp_points(good_matches_list[i], d2to3ds_list[i], structure, kps_list[i + 1])

        _, r, t, inliers = cv2.solvePnPRansac(world_pts, image_pts, K_intrinsics, np.array([]))

        # 记录相机外参
        R = cv2.Rodrigues(r)[0]
        Rots.append(R)
        trans.append(t)

        # 筛选插入点云的新三维点
        next_structure, purged_matches, new_colors = prep_next_pts(K_intrinsics, Rots[i], trans[i], R, t,
                                                       kps_list[i], kps_list[i + 1], good_matches_list[i], mass_center,
                                                       colors_list[i], config.rep_err, config.rep_per, config.mass_dist)

        if config.show_new_kp3ds:
            show_3d_points(next_structure, new_colors)

        # 点云融合
        d2to3ds_list[i], d2to3ds_list[i + 1], structure, struct_colors = fusion_structure(purged_matches,
                                                                                          d2to3ds_list[i],
                                                                                          d2to3ds_list[i + 1],
                                                                                          structure, next_structure,
                                                                                          struct_colors,
                                                                                          colors_list[i],
                                                                                          Rots[i], trans[i], R, t,
                                                                                          K_intrinsics, kps_list[i],
                                                                                          kps_list[i + 1],
                                                                                          config.rep_err)

        if config.show_raw_structure:
            show_3d_points(structure, struct_colors)

        # BA优化
        BA_model = PyBA(structure, Rots, trans, kps_list, d2to3ds_list, K_intrinsics)

        # 重投影误差求取
        residuals = BA_model.compute_residual(BA_model.total_params)

        if config.show_raw_residuals:
            plot_residuals(res.fun)
            plot_residual_distribution(res.fun)

        res, Rots, trans, structure = BA_model.run_BA()

        # 三维点云显示
        if config.show_tmp_structure:
            show_3d_points(structure, struct_colors)

        if config.show_tmp_residuals:
            plot_residuals(res.fun)
            plot_residual_distribution(res.fun)

    # 三维点云显示
    if show_final_structure:
        show_3d_points(structure, struct_colors)



