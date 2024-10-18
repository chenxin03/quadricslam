from types import TracebackType
from spatialmath import SE3
from typing import Optional, Tuple, Type, List
import numpy as np
import cv2

from ..quadricslam_states import QuadricSlamState
from . import DataSource

try:
    import pyrealsense2 as rs
except:
    raise ImportError("Failed to import 'pyrealsense2'. Please install from:"
                      "\n\thttps://pypi.org/project/pyrealsense2/")

# TODO this class currently just grabs the RGB and depth frames. For 435i
# devices there are also IMU frames (frame #2 = Accelerometer, frame #3 =
# Gyroscope). Integration could be done on this data to get a noisy odometry
# estimate.


class RealSense(DataSource):

    def __init__(self) -> None:
        # Setup the camera streams
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16,
                                  15)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8,
                                  15)
        self.pipeline = rs.pipeline()

        # Set some defaults
        self.rgb_calib = None
        self.depth_calib = None

    def __enter__(self) -> 'RealSense':
        # Start the camera
        profile = self.pipeline.start(self.config)

        # Store the camera intrinsics
        i = profile.get_stream(
            rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.rgb_calib = np.array([i.fx, i.fy, 0, i.ppx, i.ppy])

        # Get the depth scale
        self.depth_calib = float(
            profile.get_device().first_depth_sensor().get_depth_scale())
        return self

    def __exit__(self, exctype: Optional[Type[BaseException]],
                 excinst: Optional[BaseException],
                 exctb: Optional[TracebackType]) -> Optional[bool]:
        self.pipeline.stop()
        return False

    def calib_depth(self) -> float:
        if self.depth_calib is None:
            raise RuntimeError("No depth calib found. Is camera running?")
        return self.depth_calib

    def calib_rgb(self) -> np.ndarray:
        if self.rgb_calib is None:
            raise RuntimeError("No RGB calib found. Is camera running?")
        return self.rgb_calib

    def done(self) -> bool:
        return False

    def next(
        self, state: QuadricSlamState
    ) -> Tuple[Optional[SE3], Optional[np.ndarray], Optional[np.ndarray]]:
        # TODO extract odom estimate for 435i
        odom = None
        color = None
        depth = None
        while color is None or depth is None:
            fs = self.pipeline.wait_for_frames()
            color = fs.get_color_frame()
            depth = fs.get_depth_frame()

        color = np.asanyarray(color.get_data())
        depth = np.array(self.calib_depth() * np.asanyarray(depth.get_data()),
                         dtype=np.float32)
        return odom, color, depth

    def restart(self) -> None:
        pass


class SingleImageSource:
    def __init__(self, rgb_txt_path: str, depth_txt_path: str) -> None:
        # Load the paths from the txt files
        self.rgb_paths = self._load_paths(rgb_txt_path)
        self.depth_paths = self._load_paths(depth_txt_path)

        # Initialize the current index
        self.current_index = 0

        # Set some defaults
        self.rgb_calib = None
        self.depth_calib = None

    def _load_paths(self, txt_path: str) -> List[str]:
        paths = []
        with open(txt_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith('#'):
                    paths.append(line)
        return paths
    
    def __enter__(self) -> 'SingleImageSource':
        # Set up calibration parameters
        # fx = 594.21  # 水平焦距
        # fy = 591.04  # 垂直焦距
        # cx = 631.5   # 水平光心位置
        # cy = 365.4   # 垂直光心位置
        # self.rgb_calib = np.array([fx, fy, 0, cx, cy])

        self.rgb_calib = np.array([517.3, 516.5, 0, 318.6, 255.3])
        depth_scale = 0.0002  # 深度尺度
        self.depth_calib = depth_scale
        return self

    def __exit__(self, exctype: Optional[Type[BaseException]],
                 excinst: Optional[BaseException],
                 exctb: Optional[TracebackType]) -> Optional[bool]:
        return False

    def calib_depth(self) -> float:
        if self.depth_calib is None:
            raise RuntimeError("No depth calib found. Is camera running?")
        return self.depth_calib

    def calib_rgb(self) -> np.ndarray:
        if self.rgb_calib is None:
            raise RuntimeError("No RGB calib found. Is camera running?")
        return self.rgb_calib

    def done(self) -> bool:
        return self.current_index >= len(self.rgb_paths)

    def next(
        self, state: QuadricSlamState
    ) -> Tuple[Optional[SE3], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.done():
            return None, None, None

        # Load the next RGB and depth images
        rgb_path = self.rgb_paths[self.current_index]
        print(f'rgb_path = {rgb_path}')
        depth_path = self.depth_paths[self.current_index]
        print(f'depth_path = {depth_path}')

        rgb_image = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # Increment the index
        self.current_index += 1

        # Convert depth image to float32
        depth_image = depth_image.astype(np.float32) * self.calib_depth()
        return None, rgb_image, depth_image

    def restart(self) -> None:
        self.current_index = 0

    def get_num_images(self) -> int:
        return len(self.rgb_paths)



