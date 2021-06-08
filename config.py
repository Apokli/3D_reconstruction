# -*- coding: utf-8 -*-

# 文件路径名
file_route = "C:/Users/BHYBHY/Desktop/GP/objects12"

# 是否读取config.txt
have_config_txt = True

# 图片处理缩小比例
use_ratio = 2

# 粗匹配距离比例阈值
dist_coef = 0.7                 # 粗匹配距离比例阈值

# 插入点指标
mass_dist = 2                   # 空间点距点云质心距离阈值
rep_err = 15                    # 重投影误差大小阈值
rep_per = 0.03                  # 筛选掉大重投影误差比例

# 掩模绘制
draw_mask = False               # 是否绘制掩模
use_available_mask = True       # 是否用已有掩模
acquiesce_mask = True           # 是否跳过掩模查看
save_drawn_mask = False         # 是否保存掩模

# 结果显示
show_features = False           # 显示图片特征点
show_matches = False            # 显示特征点匹配对
show_init_cloud = False         # 显示初始三维点云
show_new_kp3ds = False          # 显示新加入的三维点
show_raw_structure = False      # 显示BA优化前三维点云
show_raw_residuals = False      # 显示BA优化前重投影误差分布
show_tmp_structure = False      # 显示BA优化后三维点云
show_tmp_residuals = False      # 显示BA优化后重投影误差分布
show_final_structure = True     # 显示最终三维点云
