from itertools import groupby
from types import FunctionType
from typing import Callable, Dict, List, Optional, Union
import cv2

import gtsam
import gtsam_quadrics
import numpy as np
from spatialmath import SE3

from .data_associator import DataAssociator
from .data_source import DataSource
from .detector import Detector
from .quadricslam_states import QuadricSlamState, StepState, SystemState
from .utils import (
    QuadricInitialiser,
    initialise_quadric_ray_intersection,
    new_factors,
    new_values,
)
from .visual_odometry import VisualOdometry


class QuadricSlam:

    def __init__(
        self,
        data_source: DataSource,
        visual_odometry: Optional[VisualOdometry] = None,
        detector: Optional[Detector] = None,
        associator: Optional[DataAssociator] = None,
        initial_pose: Optional[SE3] = None,
        noise_prior: np.ndarray = np.array([0] * 6, dtype=np.float64),
        noise_odom: np.ndarray = np.array([0.01] * 6, dtype=np.float64),
        noise_boxes: np.ndarray = np.array([3] * 4, dtype=np.float64),
        optimiser_batch: Optional[bool] = None,
        optimiser_params: Optional[Union[gtsam.ISAM2Params,
                                         gtsam.LevenbergMarquardtParams,
                                         gtsam.GaussNewtonParams]] = None,
        on_new_estimate: Optional[Callable[[QuadricSlamState], None]] = None,
        quadric_initialiser:
        QuadricInitialiser = initialise_quadric_ray_intersection
    ) -> None:
        # TODO this needs a default data associator, we can't do anything
        # meaningful if this is None...
        # if associator is None:
        #     raise NotImplementedError('No default data associator yet exists, '
        #                               'so you must provide one.')
        self.associator = associator
        self.data_source = data_source
        self.detector = detector
        self.visual_odometry = visual_odometry

        self.on_new_estimate = on_new_estimate
        self.quadric_initialiser = quadric_initialiser

        # Bail if optimiser settings and modes aren't compatible
        if (optimiser_batch == True and
                type(optimiser_params) == gtsam.ISAM2Params):
            raise ValueError("ERROR: Can't run batch mode with '%s' params." %
                             type(optimiser_params))
        elif (optimiser_batch == False and optimiser_params is not None and
              type(optimiser_params) != gtsam.ISAM2Params):
            raise ValueError(
                "ERROR: Can't run incremental mode with '%s' params." %
                type(optimiser_params))
        if optimiser_params is None:
            optimiser_params = (gtsam.LevenbergMarquardtParams()
                                if optimiser_batch is True else
                                gtsam.ISAM2Params())

        # Setup the system state, and perform a reset
        self.state = QuadricSlamState(
            SystemState(
                initial_pose=SE3() if initial_pose is None else initial_pose,
                noise_prior=noise_prior,
                noise_odom=noise_odom,
                noise_boxes=noise_boxes,
                optimiser_batch=type(optimiser_params) != gtsam.ISAM2Params,
                optimiser_params=optimiser_params))
        self.reset()

    def guess_initial_values(self) -> None:
        # 猜测方法（只对没有估计值的变量进行猜测）:
        # - 使用航位推算猜测位姿
        # - 使用所有观测数据的欧几里得平均值猜测二次曲面
        s = self.state.system

        # 获取图中所有因子
        fs = [s.graph.at(i) for i in range(0, s.graph.nrFactors())]
        
        # 没有估计值的位姿添加估计值
        for pf in [
                f for f in fs if type(f) == gtsam.PriorFactorPose3 and
                not s.estimates.exists(f.keys()[0])
        ]:
            s.estimates.insert(pf.keys()[0], pf.prior())

        # 逐个添加相对因子，剩余因子迁移至原点
        bfs = [f for f in fs if type(f) == gtsam.BetweenFactorPose3]
        done = False
        while not done:
            bf = next((f for f in bfs if s.estimates.exists(f.keys()[0]) and
                       not s.estimates.exists(f.keys()[1])), None)
            if bf is None:
                done = True
                continue
            s.estimates.insert(
                bf.keys()[1],
                s.estimates.atPose3(bf.keys()[0]) * bf.measured())
            bfs.remove(bf)
        for bf in [
                f for f in bfs if not all([
                    s.estimates.exists(f.keys()[i])
                    for i in range(0, len(f.keys()))
                ])
        ]:
            s.estimates.insert(bf.keys()[1], gtsam.Pose3())

        # 将所有的物体因子按objectkey排序
        _ok = lambda x: x.objectKey()
        bbs = sorted([
                    f for f in fs if type(f) == gtsam_quadrics.BoundingBoxFactor and
                    not s.estimates.exists(f.objectKey())],
                    key=_ok)
        # 按objectkey分组，为所有物体初始化二次曲面
        for qbbs in [list(v) for k, v in groupby(bbs, _ok)]:
            self.quadric_initialiser(
                [s.estimates.atPose3(bb.poseKey()) for bb in qbbs],
                [bb.measurement() for bb in qbbs],
                self.state).addToValues(s.estimates, qbbs[0].objectKey())

    def spin(self, return_s: bool=False) -> None:
        while not self.data_source.done():
            self.step()

        if self.state.system.optimiser_batch:
            self.guess_initial_values()
            s = self.state.system
            s.optimiser = s.optimiser_type(s.graph, s.estimates,
                                           s.optimiser_params)
            s.estimates = s.optimiser.optimize()
        if self.on_new_estimate:
            self.state.system.optimiser_batch = True
            self.on_new_estimate(self.state)

        if return_s:
            return self.state

    def step(self) -> None:
        # 设置当前步骤的状态
        s = self.state.system
        p = self.state.prev_step
        n = StepState(
            0 if self.state.prev_step is None else self.state.prev_step.i + 1)
        self.state.this_step = n

        # 从场景中获取最新数据（里程计、图像及检测结果）
        n.odom, n.rgb, n.depth = (self.data_source.next(self.state))
        # 数据源没有传递位姿时，使用位姿里程计计算
        if n.odom is None:
            if self.visual_odometry is not None:
                n.odom = self.visual_odometry.odom(self.state)
        # 物体检测
        n.detections = (self.detector.detect(self.state)
                        if self.detector else [])
        if self.associator is not None:
            # 关联器筛选新增物体
            n.new_associated, s.associated, s.unassociated = (self.associator.associate(self.state))
        else:
            n.new_associated = n.detections
            s.associated = n.detections
        print(f'新增关联：{len(n.new_associated)} 个')
 
        for i in range(len(s.associated)):
            x1, y1, x2, y2 = s.associated[i].bounds
            depth_ = n.depth[int(y1) : int(y2), int(x1) : int(x2)]
            depth_mean = np.mean(depth_)
            s.associated[i].depth_mean = depth_mean
        
        # 提取一些标签信息
        # TODO 处理单个二次曲面使用不同标签的情况
        s.labels = {
            d.quadric_key: d.label + ' depth:' + str(d.depth_mean)
            for d in s.associated
            if d.quadric_key is not None
        }

        # 将新姿态添加到因子图中
        if p is None:
            s.graph.add(
                gtsam.PriorFactorPose3(n.pose_key, s.initial_pose,
                                    s.noise_prior))
        else:
            s.graph.add(
                gtsam.BetweenFactorPose3(
                    p.pose_key, n.pose_key,
                    gtsam.Pose3(((SE3() if p.odom is None else p.odom).inv() *
                                (SE3() if n.odom is None else n.odom)).A),
                    s.noise_odom))

        # 将新关联的检测结果添加到因子图中
        for d in n.new_associated:
            if d.quadric_key is None:
                print("WARN: 跳过具有 None 类型 quadric_key 的关联检测")
                continue
            s.graph.add(
                gtsam_quadrics.BoundingBoxFactor(
                    gtsam_quadrics.AlignedBox2(d.bounds),
                    gtsam.Cal3_S2(s.calib_rgb), d.pose_key, d.quadric_key,
                    s.noise_boxes))

        # 如果处于迭代模式，则进行优化
        if not s.optimiser_batch:
            self.guess_initial_values()
            if s.optimiser is None:
                s.optimiser = s.optimiser_type(s.optimiser_params)
            try:
                # print(f'optimiser set {s.optimiser}')
                # pu.db
                s.optimiser.update(
                    new_factors(s.graph, s.optimiser.getFactorsUnsafe()),
                    new_values(s.estimates,
                            s.optimiser.getLinearizationPoint()))
                s.estimates = s.optimiser.calculateEstimate()
            except RuntimeError as e:
                # 处理 gtsam::IndeterminateLinearSystemException 异常
                pass
            if self.on_new_estimate:
                self.on_new_estimate(self.state)
            
        self.state.prev_step = n


    def reset(self) -> None:
        self.data_source.restart()

        s = self.state.system
        s.associated = []
        s.unassociated = []
        s.labels = {}
        s.graph = gtsam.NonlinearFactorGraph()
        s.estimates = gtsam.Values()
        s.optimiser = (None if s.optimiser_batch else s.optimiser_type(
            s.optimiser_params))

        s.calib_depth = self.data_source.calib_depth()
        s.calib_rgb = self.data_source.calib_rgb()

        self.state.prev_step = None
        self.state.this_step = None
