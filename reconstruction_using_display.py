# -*- coding: utf-8 -*-

import reconstruction_display
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QRect, QThread, pyqtSignal
import sys
import numpy as np
import cv2
from main_functions import *
from small_support_functions import *
from display_functions import *
from bundle_adjustment import PyBA
import time

display = reconstruction_display.Ui_main_window


class Reconstruction(QMainWindow, display):

    stop_thread = pyqtSignal(bool)

    def __init__(self):
        QMainWindow.__init__(self)
        display.__init__(self)

        self.setupUi(self)
        self.image_display_size = (self.processing_image.height(), self.processing_image.width())

        self.stop = False

        self.kps_list = []
        self.resized_kps_list = []
        self.descs_list = []
        self.colors_list = []
        self.resize_ratio = 0
        self.image_list = []
        self.image_name_list = []
        self.image_complete_name_list = []
        self.resized_image_list = []

        self.button_functioning()

    def button_functioning(self):
        self.start_reconstruction_button.clicked.connect(self.on_start_reconstruction)
        self.end_reconstruction_button.clicked.connect(self.on_end_reconstruction)

    def on_start_reconstruction(self):
        file_route = self.file_route_text.toPlainText()
        self.get_photo_thread = GetPhotoThread(file_route, self.image_display_size)
        self.get_photo_thread.transmit_photos.connect(self.on_receive_photos)
        self.stop_thread.connect(self.get_photo_thread.handle_stop)
        self.get_photo_thread.start()

    def on_receive_photos(self, things):
        if not self.stop:
            if isinstance(things, int):
                if things == 0:
                    self.information_text.insertPlainText("文件路径有误，请检查文件路径！\n")
                elif things == -1:
                    self.information_text.insertPlainText("重建已终止\n")
            else:
                self.K = things[0]
                self.resize_ratio = things[1]
                self.image_list = things[2]
                self.image_name_list = things[3]
                self.image_complete_name_list = things[4]
                self.resized_image_list = things[5]
                self.total_size = len(self.image_list)
                self.information_text.insertPlainText("图片读取完成！\n")
                self.information_text.insertPlainText("共有：" + str(len(self.image_list)) + "张图片\n")
                if self.total_size < 2:
                    self.information_text.insertPlainText("图片数目不足(<2)，无法进行三维重建\n")
                    return
                else:
                    self.information_text.insertPlainText(self.image_name_list[0] + ", " + self.image_name_list[1])
                    if self.total_size > 2:
                        self.information_text.insertPlainText(", ...\n")
                self.feature_thread = FeatureThread(self.image_list, self.resize_ratio, self.resized_image_list,
                                                    self.image_complete_name_list)
                self.feature_thread.transmit_features.connect(self.on_receive_features)
                self.stop_thread.connect(self.feature_thread.handle_stop)
                self.feature_thread.start()
        else:
            self.information_text.insertPlainText("重建已终止\n")

    def on_receive_features(self, things):
        if isinstance(things, int):
            if things == -1:
                self.information_text.insertPlainText("重建已终止\n")
        else:
            self.kps_list.append(things[0][0])
            self.resized_kps_list.append(things[1][0])
            self.descs_list.append(things[2][0])
            self.colors_list.append(things[3][0])
            keypoint_image = things[4]
            num = things[5]
            show_image = QImage(keypoint_image, keypoint_image.shape[1], keypoint_image.shape[0], keypoint_image.shape[1] * 3, QImage.Format_RGB888)
            self.processing_image.drawPixmap(QRect((self.image_display_size[1] - keypoint_image.shape[1]) // 2,
                                                   (self.image_display_size[0] - keypoint_image.shape[0]) // 2,
                                                   keypoint_image.shape[1], keypoint_image.shape[0]), QPixmap(show_image))
            self.information_text.insertPlainText("已成功提取第" + str(num) + "个图片特征，共：" + str(len(things[0][0])) + "\n")
            if things[6] == 1:
                self.information_text.insertPlainText("特征提取完毕，开始匹配\n")

    def on_end_reconstruction(self):
        self.stop = True
        self.stop_thread.emit(True)
        self.information_text.insertPlainText("重建正在终止...\n")


class GetPhotoThread(QThread):
    transmit_photos = pyqtSignal(object)

    def __init__(self, file_route, image_display_size):
        super().__init__()
        self.file_route = file_route
        self.image_display_size = image_display_size
        self.stop = False

    def __del__(self):
        self.wait()

    def handle_stop(self):
        self.stop = True

    def run(self):
        try:
            if not self.stop:
                resize_ratio, image_list, image_name_list, \
                image_complete_name_list, resized_image_list = get_photos(file_route, self.image_display_size)
                K = get_K(file_route)
                self.transmit_photos.emit((K, resize_ratio, image_list, image_name_list, image_complete_name_list, resized_image_list))
            else:
                self.transmit_photos.emit(-1)
        except:
            self.transmit_photos.emit(0)


class FeatureThread(QThread):
    transmit_features = pyqtSignal(object)

    def __init__(self, image_list, resize_ratio, resized_image_list, image_complete_name_list):
        super().__init__()
        self.stop = False
        self.image_list = image_list
        self.resize_ratio = resize_ratio
        self.resized_image_list = resized_image_list
        self.image_complete_name_list = image_complete_name_list
        self.total_size = len(image_list)

    def __del__(self):
        self.wait()

    def handle_stop(self):
        self.stop = True

    def run(self):
        for i in range(self.total_size):
            if not self.stop:
                rimg = self.resized_image_list[i]
                kps, resized_kps, descs, colors = extract_features([self.image_list[i]], self.resize_ratio,
                                                                   [rimg], [self.image_complete_name_list[i]])
                keypoint_img = show_keypoints(rimg, resized_kps[0])
                if i != self.total_size - 1:
                    self.transmit_features.emit((kps, resized_kps, descs, colors, keypoint_img, i, 0))
                else:
                    self.transmit_features.emit((kps, resized_kps, descs, colors, keypoint_img, i, 1))
            else:
                self.transmit_features.emit(-1)
                break


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Reconstruction()
    window.show()
    sys.exit(app.exec_())
