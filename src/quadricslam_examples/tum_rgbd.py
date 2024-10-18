#!/usr/bin/env python3

import inspect
import numpy as np
import os
import spatialmath as sm
import sys
import textwrap

from quadricslam.data_associator.quadric_iou_associator import QuadricIouAssociator
from quadricslam import QuadricSlam, visualise, utils
from quadricslam.data_source.tum_rgbd import TumRgbd
from quadricslam.detector.faster_rcnn import FasterRcnn
from quadricslam.detector.yolov8 import YoloV8Detector

import pudb


def run():
    # Print a verbose note each time this is run about getting the dataset
    print(
        "%s\n\nThe script can be run via:\n\t%s DESTINATION\n" %
        (textwrap.fill(
            "NOTE: This example requires the path to an already downloaded TUM "
            "RGBD dataset to be provided as the first argument. If you don't "
            "have the datasets downloaded, we have a helper script with the "
            "data source to get the datasets for you.",
            width=80),
         os.path.join(os.path.dirname(inspect.getfile(TumRgbd)),
                      'get_tum_rgbd_datasets')))

    # Confirm dataset path is provided
    if len(sys.argv) != 2:
        print("ERROR: Path to dataset is a required argument.")
        sys.exit(1)
    dataset_path = sys.argv[1]

    # Pull camera calibration parameters from:
    #  https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats#intrinsic_camera_calibration_of_the_kinect
    camera_calib = np.array(
        [517.3, 516.5, 0, 318.6, 255.3] if 'freiburg1' in
        dataset_path else [520.9, 521.0, 0, 325.1, 249.7] if 'freiburg2' in
        dataset_path else [535.4, 539.2, 0, 320.1, 247.6] if 'freiburg3' in
        dataset_path else [525.0, 525.0, 0, 319.5, 239.5])

    # Run QuadricSLAM
    q = QuadricSlam(
        data_source=TumRgbd(path=dataset_path, rgb_calib=camera_calib, depth_calib=5000),
        detector=YoloV8Detector(model_path='/home/cnbot/zz/slam/yolov8m.pt', detection_thresh=0.7,is_show=True),
        # TODO needs a viable data association approach
        associator=QuadricIouAssociator(iou_thresh=0.05),
        on_new_estimate=(lambda state: visualise(state.system.estimates, state.system.labels, state.system.optimiser_batch)),
        quadric_initialiser=utils.initialise_quadric_from_depth
        )
    q.spin()


if __name__ == '__main__':
    run()
