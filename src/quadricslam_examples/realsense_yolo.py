#!/usr/bin/env python

from quadricslam import QuadricSlam, utils

from quadricslam.data_associator.quadric_iou_associator import QuadricIouAssociator
from quadricslam.data_source.realsense import SingleImageSource
from quadricslam.data_source.tum_rgbd_fix import TumRgbd
from quadricslam.detector.yolov8 import YoloV8Detector
from quadricslam.visual_odometry.rgbd_cv2 import RgbdCv2
from quadricslam.visualisation import visualise
from quadricslam.utils import ps_and_qs_from_values

import numpy as np
import time
import os


def run():
    # print('start run')
    
    # with SingleImageSource(rgb_txt_path=rgb_txt_path, depth_txt_path=depth_txt_path) as ds:
    q = QuadricSlam(
        # 数据读取模块
        data_source = TumRgbd(dataset_path='/home/cnbot/zz/slam/datasets/rgbd_dataset_freiburg1_desk', step=2),        
        # 目标检测模块
        detector = YoloV8Detector(model_path='/home/cnbot/zz/slam/yolov8m.pt', detection_thresh=0.7, is_save=False, is_show=True),
        # 视觉里程计     
        visual_odometry=RgbdCv2(),    
        # 关联器，推断每一帧检测目标与上一帧目标的关联性，判断是否为同一目标
        associator = QuadricIouAssociator(iou_thresh=0.05),    
        # 是否使用批处理  
        optimiser_batch = True,      
        # on_new_estimate = (lambda state: visualise_draw(state.this_step.rgb, state.system.estimates, state.system.labels)),
        on_new_estimate = (lambda state: visualise(values=state.system.estimates, labels=state.system.labels, block=state.system.optimiser_batch)),
        quadric_initialiser = utils.initialise_quadric_ray_intersection)
    # print('q init')
    state = q.spin(return_s=True)    
    utils.save_state(state, save_img=True)

if __name__ == '__main__':
    run()
