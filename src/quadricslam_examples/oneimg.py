#!/usr/bin/env python

from quadricslam import QuadricSlam, utils

from quadricslam.data_associator.quadric_iou_associator import DefaultAssociator
from quadricslam.data_source.OneImg import OneImg
from quadricslam.detector.yolov8 import YoloV8Detector
from quadricslam.visual_odometry.rgbd_cv2 import RgbdCv2
from quadricslam.visualisation import visualise
from quadricslam.utils import ps_and_qs_from_values



def run():
    q = QuadricSlam(
        # 数据读取模块
        data_source = OneImg(dataset_path='/home/cnbot/zz/slam/datasets/one2', step=1),        
        # 目标检测模块
        detector = YoloV8Detector(model_path='/home/cnbot/zz/slam/yolov8m.pt', detection_thresh=0.7, is_save=False, is_show=False), 
        # 视觉里程计     
        visual_odometry=RgbdCv2(), 
        # 关联器
        associator = DefaultAssociator(),  
        # 是否使用批处理  
        optimiser_batch = True,      
        on_new_estimate = (lambda state: visualise(values=state.system.estimates, labels=state.system.labels, block=state.system.optimiser_batch)),
        quadric_initialiser = utils.initialise_quadric_from_depth)
    state = q.spin(return_s=True)    
    utils.save_state(state, save_img=True)

if __name__ == '__main__':
    run()
