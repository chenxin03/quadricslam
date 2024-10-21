from spatialmath import SE3, UnitQuaternion
from typing import List, Optional, Tuple
import cv2
import numpy as np
import os

from ..quadricslam_states import QuadricSlamState
from . import DataSource

class OneImg(DataSource):
    """
    单图数据源
    """
    def __init__(self, 
                dataset_path: str,
                rgb_calib: np.ndarray = [517.3, 516.5, 0, 318.6, 255.3], 
                depth_calib: float = 5000,
                step: int = 1) -> None:
        
        self.rgb_calib = rgb_calib
        self.depth_calib = depth_calib

        self.rgb_image = cv2.imread(os.path.join(dataset_path, 'rgb.png'))
        self.depth_image = cv2.imread(os.path.join(dataset_path, 'depth.png'), cv2.IMREAD_UNCHANGED)
        gt = open(os.path.join(dataset_path, 'groundtruth.txt'))
        self.groundtruth = [float(x) for x in gt.readline().strip().split()]
        self.step = step
        self.restart()

    def next(self, state: QuadricSlamState) -> Tuple[Optional[SE3], Optional[np.ndarray], Optional[np.ndarray]]:
        self.i += 1
        return (
                # None,
                self._gt_to_SE3(),
                self.rgb_image, 
                # None
                # 增加对depth的读取
                self.depth_image.astype(np.float32) / self.calib_depth()
                )
    
    def calib_rgb(self) -> np.ndarray:
        if self.rgb_calib is None:
            raise RuntimeError("No RGB calib found. Is camera running?")
        return self.rgb_calib

    def calib_depth(self) -> np.ndarray:
        if self.depth_calib is None:
            raise RuntimeError("No Depth calib found. Is camera running?")
        return self.depth_calib

    def _gt_to_SE3(self) -> SE3:
        f = np.asarray(self.groundtruth, float)
        return SE3.Rt(UnitQuaternion(f[6], f[3:6]).SO3(), f[0:3])

    def done(self) -> bool:
        return self.i >= self.step

    def restart(self) -> None:
        self.i = 0