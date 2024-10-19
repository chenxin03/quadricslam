#!/usr/bin/env python

from quadricslam import QuadricSlam, utils, quadricslam_states

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


def save_state(state: quadricslam_states.QuadricSlamState) -> None:
    values = state.system.estimates
    labels = state.system.labels
    ps, qs = ps_and_qs_from_values(values)
    # print(f'ps:{ps}')
    # print(f'qs:{qs}')
    poses_cap = [p.matrix() for p in ps.values()]
    poses_obj = [q.pose().matrix() for _, q in qs.items()]
    radius = [q.radii() for _, q in qs.items()]
    keys = [key for key, _ in qs.items()]
    timesnap = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))
    base_path = os.path.expanduser(os.path.join('~/zz/slam/quadricslam/results/', timesnap))
    if not os.path.exists(os.path.expanduser(base_path)):
        os.makedirs(os.path.expanduser(base_path))


    with open(os.path.join(base_path, f'CapturePose.csv'), 'w') as c:
        c.write("pose_cap_00,pose_cap_01,pose_cap_02,pose_cap_03,"
                "pose_cap_10,pose_cap_11,pose_cap_12,pose_cap_13,"
                "pose_cap_20,pose_cap_21,pose_cap_22,pose_cap_23,"
                "pose_cap_30,pose_cap_31,pose_cap_32,pose_cap_33\n")
        for pose_cap in poses_cap:
            cap_flat = pose_cap.flatten()
            c.write(",".join(map(str, cap_flat)) +"\n")

    with open(os.path.join(base_path, f'Object.csv'), 'w') as o:
        o.write("pose_object_00,pose_object_01,pose_object_02,pose_object_03,"
                "pose_object_10,pose_object_11,pose_object_12,pose_object_13,"
                "pose_object_20,pose_object_21,pose_object_22,pose_object_23,"
                "pose_object_30,pose_object_31,pose_object_32,pose_object_33,"
                "radius_x,radius_y,radius_z,label\n")
        for pose_obj, radiu, key in zip(poses_obj, radius, keys):
            obj_flat = pose_obj.flatten()
            radiu_flat = radiu.flatten()
            label = str(labels[key])
            o.write(",".join(map(str, obj_flat)) + "," + ",".join(map(str, radiu_flat)) + "," + label + "\n")
    print(f'数据已写入：{base_path}')

def run():
    # print('start run')
    
    # with SingleImageSource(rgb_txt_path=rgb_txt_path, depth_txt_path=depth_txt_path) as ds:
    q = QuadricSlam(
        # 数据读取模块
        data_source = TumRgbd(dataset_path='/home/cnbot/zz/slam/datasets/rgbd_dataset_freiburg1_desk', step=5),        
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
    save_state(state)

if __name__ == '__main__':
    run()
