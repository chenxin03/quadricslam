from typing import List

from ..quadricslam_states import Detection, QuadricSlamState
from . import Detector

from ultralytics import YOLO

class YoloV8Detector(Detector):
    """
    使用YOLOv8模型进行目标检测。
    """

    def __init__(
            self, model_path: str = 'yolov8n.pt', 
            detection_thresh: float = 0.5,
            is_save: bool = False,
            is_show: bool = False) -> None:
        """
        初始化YOLOv8模型。

        :param model_path: 模型路径或名称
        :param detection_thresh: 检测阈值
        """
        self.model = YOLO(model_path)
        self.detection_thresh = detection_thresh
        self.is_show = is_show
        self.is_save = is_save

    def detect(self, state: QuadricSlamState) -> List[Detection]:
        """
        对给定的状态执行检测。

        :param state: 包含图像的状态对象
        :return: 检测结果列表
        """
        assert state.this_step is not None
        assert state.this_step.rgb is not None
        n = state.this_step

        # 使用YOLOv8模型进行预测
        results = self.model(n.rgb, show=self.is_show, save=self.is_save, conf=self.detection_thresh)

        # 解析结果
        detections = []
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > self.detection_thresh:
                # 假设类别ID与类名之间的映射存储在self.model.names中
                label = self.model.names[int(class_id)]
                detections.append(Detection(label=label, 
                                            bounds=[x1, y1, x2, y2], 
                                            pose_key=n.pose_key))

        return detections