#!/usr/bin/env python

from quadricslam import QuadricSlam, utils

from quadricslam.data_associator.quadric_iou_associator import QuadricIouAssociator
from quadricslam.data_source.realsense import SingleImageSource
from quadricslam.data_source.tum_rgbd_fix import TumRgbd
from quadricslam.detector.yolov8 import YoloV8Detector
from quadricslam.visual_odometry.rgbd_cv2 import RgbdCv2
from quadricslam.visualisation import visualise

import numpy as np


rgb_txt_path = '/home/cnbot/zz/slam/datasets/rgb_path.txt'
depth_txt_path = '/home/cnbot/zz/slam/datasets/depth_path.txt'

def run():
    # print('start run')
    ds = TumRgbd(dataset_path='/home/cnbot/zz/slam/datasets/rgbd_dataset_freiburg1_rpy')
    # with SingleImageSource(rgb_txt_path=rgb_txt_path, depth_txt_path=depth_txt_path) as ds:
    q = QuadricSlam(
        # 数据读取模块
        data_source=ds,        
        # 目标检测模块
        detector=YoloV8Detector(model_path='/home/cnbot/zz/slam/yolov8m.pt', detection_thresh=0.7,is_show=True),
        # 视觉里程计     
        visual_odometry=RgbdCv2(),    
        # 关联器，推断每一帧检测目标与上一帧目标的关联性，判断是否为同一目标  
        associator=QuadricIouAssociator(iou_thresh=0.05),    
        # 是否使用批处理  
        optimiser_batch=False,      
        # on_new_estimate=(lambda state: visualise_draw(state.this_step.rgb, state.system.estimates, state.system.labels)),
        on_new_estimate=(lambda state: visualise(state.system.estimates, state.system.labels, state.system.optimiser_batch)),
        quadric_initialiser=utils.initialise_quadric_from_depth)
    # print('q init')
    q.spin()    


if __name__ == '__main__':
    run()
