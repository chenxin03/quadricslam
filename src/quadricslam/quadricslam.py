from itertools import groupby
from types import FunctionType
from spatialmath import SE3
from typing import Callable, Dict, List, Optional, Union
import gtsam
import gtsam_quadrics
import numpy as np

from .data_associator import DataAssociator
from .data_source import DataSource
from .detector import Detector
from .quadricslam_states import QuadricSlamState, StepState, SystemState
from .visual_odometry import VisualOdometry

import pudb


def guess_quadric(
        obs_poses: List[gtsam.Pose3], boxes: List[gtsam_quadrics.AlignedBox2],
        calib: gtsam.Cal3_S2) -> gtsam_quadrics.ConstrainedDualQuadric:
    # TODO replace with something less dumb...

    # Get each observation point
    ps = np.array([op.translation() for op in obs_poses])

    # Get each observation direction
    # TODO actually use bounding box rather than assuming middle...
    vs = np.array([op.rotation().matrix()[:, 0] for op in obs_poses])

    # Apply this to compute point closet to where all rays converge:
    #   https://stackoverflow.com/a/52089698/1386784
    i_minus_vs = np.eye(3) - (vs[:, :, np.newaxis] @ vs[:, np.newaxis, :])
    quadric_centroid = np.linalg.lstsq(
        i_minus_vs.sum(axis=0),
        (i_minus_vs @ ps[:, :, np.newaxis]).sum(axis=0),
        rcond=None)[0].squeeze()

    # Fudge the rest for now
    # TODO do better...
    return gtsam_quadrics.ConstrainedDualQuadric(
        gtsam.Rot3(), gtsam.Point3(quadric_centroid), [1, 1, 0.1])


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
        on_new_estimate: Optional[Callable[[QuadricSlamState], None]] = None
    ) -> None:
        # TODO this needs a default data associator, we can't do anything
        # meaningful if this is None...
        if associator is None:
            raise NotImplementedError('No default data associator yet exists, '
                                      'so you must provide one.')
        self.associator = associator
        self.data_source = data_source
        self.detector = detector
        self.visual_odometry = visual_odometry

        self.on_new_estimate = on_new_estimate

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
                                if optimiser_batch is not False else
                                gtsam.ISAM2Params())

        # Setup the system state, and perform a reset
        self.state = QuadricSlamState(
            SystemState(
                initial_pose=SE3() if initial_pose is None else initial_pose,
                noise_prior=noise_prior,
                noise_odom=noise_odom,
                noise_boxes=noise_boxes,
                optimiser_batch=type(optimiser_params) != gtsam.ISAM2,
                optimiser_params=optimiser_params))
        self.reset()

    def guess_initial_values(self) -> None:
        # Guessing approach (only guess values that don't already have an
        # estimate):
        # - guess poses using dead reckoning
        # - guess quadrics using Euclidean mean of all observations
        s = self.state.system

        fs = [s.graph.at(i) for i in range(0, s.graph.nrFactors())]

        # Start with prior factors
        for pf in [
                f for f in fs if type(f) == gtsam.PriorFactorPose3 and
                not s.estimates.exists(f.keys()[0])
        ]:
            s.estimates.insert(pf.keys()[0], pf.prior())

        # Add all between factors one-by-one (should never be any remaining,
        # but if they are just dump them at the origin after the main loop)
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
        for bf in bfs:
            s.estimates.insert(bf.keys()[1], gtsam.Pose3())

        # Add all quadric factors
        _ok = lambda x: x.objectKey()
        bbs = sorted([
            f for f in fs if type(f) == gtsam_quadrics.BoundingBoxFactor and
            not s.estimates.exists(f.objectKey())
        ],
                     key=_ok)
        for qbbs in [list(v) for k, v in groupby(bbs, _ok)]:
            guess_quadric([s.estimates.atPose3(bb.poseKey()) for bb in qbbs],
                          [bb.measurement for bb in qbbs],
                          gtsam.Cal3_S2(
                              self.data_source.calib_rgb())).addToValues(
                                  s.estimates, qbbs[0].objectKey())

    def spin(self) -> None:
        while not self.data_source.done():
            self.step()

        if self.state.system.optimiser_batch:
            self.guess_initial_values()
            s = self.state.system
            s.optimiser = s.optimiser_type(s.graph, s.estimates,
                                           s.optimiser_params)
            s.estimates = s.optimiser.optimize()

        if self.on_new_estimate:
            self.on_new_estimate(self.state)

    def step(self) -> None:
        # Setup state for the current step
        s = self.state.system
        p = self.state.prev_step
        n = StepState(
            0 if self.state.prev_step is None else self.state.prev_step.i + 1)
        self.state.this_step = n

        # Get latest data from the scene (odom, images, and detections)
        n.odom, n.rgb, n.depth = (self.data_source.next(self.state))
        if self.visual_odometry is not None:
            n.odom = self.visual_odometry.odom(self.state)
        n.detections = (self.detector.detect(self.state)
                        if self.detector else [])
        n.new_associated, s.associated, s.unassociated = (
            self.associator.associate(self.state))

        # Extract some labels
        # TODO handle cases where different labels used for a single quadric???
        s.labels = {
            d.quadric_key: d.label
            for d in s.associated
            if d.quadric_key is not None
        }

        # # Add new pose to the factor graph
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

        # Add any newly associated detections to the factor graph
        for d in n.new_associated:
            s.graph.add(
                gtsam_quadrics.BoundingBoxFactor(
                    gtsam_quadrics.AlignedBox2(d.bounds),
                    gtsam.Cal3_S2(s.calib_rgb), d.pose_key, d.quadric_key,
                    s.noise_boxes))

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
