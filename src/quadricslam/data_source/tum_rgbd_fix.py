from spatialmath import SE3, UnitQuaternion
from subprocess import check_output
from typing import List, Optional, Tuple, Union, cast
import cv2
import numpy as np
import os

from ..quadricslam_states import QuadricSlamState
from . import DataSource


class TumRgbd(DataSource):
    def __init__(self, dataset_path: str, 
                 rgb_calib: np.ndarray = [517.3, 516.5, 0, 318.6, 255.3], 
                 depth_calib: float = 5000,
                 step: int = 0) -> None:
        # Validate path exists
        self.base_path = dataset_path
        if not os.path.isdir(self.base_path):
            raise ValueError("Path '%s' does not exist." % self.base_path)

        # Store camera calibration
        self.rgb_calib = rgb_calib
        self.depth_calib = depth_calib

        # Derive synced dataset (aligning on depth as it always has the least
        # data)
        self._load_data()
        if step == 0:
            self.step = self.data_length
        else:
            self.step = step
        self.restart()

    def _load_data(self):
        data_file_path = os.path.join(self.base_path, 'data.txt')

        with open(data_file_path, 'r') as f:
            lines = f.readlines()

        def parse_line(line):
            parts = line.strip().split()
            return {
                'rgb_time': float(parts[0]),
                'rgb_path': parts[1],
                'depth_time': float(parts[2]),
                'depth_path': parts[3],
                'groundtruth_time': float(parts[4]),
                'groundtruth': list(map(float, parts[5:]))
            }
        
        data = [parse_line(line) for line in lines]

        depth_list = [item['depth_path'] for item in data]
        rgb_list = [item['rgb_path'] for item in data]
        groundtruth_list = [item['groundtruth'] for item in data]

        self.data = {
            'depth': depth_list,
            'rgb': rgb_list,
            'groundtruth': groundtruth_list
            }
        self.data_length = len(self.data['depth'])


    def next(self) -> Tuple[Optional[SE3], Optional[np.ndarray], Optional[np.ndarray]]:
        i = self.data_i
        self.data_i += 1

        rgb_image = cv2.imread(os.path.join(self.base_path, self.data['rgb'][i]))
        depth_image = cv2.imread(os.path.join(self.base_path, self.data['depth'][i]), cv2.IMREAD_UNCHANGED)

        return (
                # None,
                SE3() if i == 0 else self._gt_to_SE3(i) * self._gt_to_SE3(i - 1).inv(),
                rgb_image, 
                # None
                # 增加对depth的读取
                depth_image.astype(np.float32) / self.calib_depth()
                )
    
    def calib_rgb(self) -> np.ndarray:
        if self.rgb_calib is None:
            raise RuntimeError("No RGB calib found. Is camera running?")
        return self.rgb_calib

    def calib_depth(self) -> np.ndarray:
        if self.depth_calib is None:
            raise RuntimeError("No Depth calib found. Is camera running?")
        return self.depth_calib

    def _gt_to_SE3(self, i: int) -> SE3:
        f = np.asarray(self.data['groundtruth'][i], float)
        return SE3.Rt(UnitQuaternion(f[6], f[3:6]).SO3(), f[0:3])

    def done(self) -> bool:
        return self.data_i == self.step

    
    def restart(self) -> None:
        self.data_i = 0
