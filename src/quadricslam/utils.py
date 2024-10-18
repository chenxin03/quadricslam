from typing import Callable, List
import gtsam
import gtsam_quadrics
import numpy as np

from .quadricslam_states import QuadricSlamState

QuadricInitialiser = Callable[
    [List[gtsam.Pose3], List[gtsam_quadrics.AlignedBox2], QuadricSlamState],
    gtsam_quadrics.ConstrainedDualQuadric]


def initialise_quadric_from_depth(
        obs_poses: List[gtsam.Pose3],
        boxes: List[gtsam_quadrics.AlignedBox2],
        state: QuadricSlamState,
        object_depth=0.1) -> gtsam_quadrics.ConstrainedDualQuadric:
    # 使用深度图像从单视角初始化四次曲面（注意：
    # 这假设只有一个视角，并且将丢弃所有额外的视角）
    s = state.system
    assert s.calib_rgb is not None
    assert state.this_step is not None
    n = state.this_step
    assert n.depth is not None

    box = boxes[0]
    pose = obs_poses[0]
    calib = gtsam.Cal3_S2(s.calib_rgb)

    # 获取平均的包围框深度
    dbox = box.vector().astype('int')  # 获取离散的包围框边界
    box_depth = n.depth[dbox[1]:dbox[3], dbox[0]:dbox[2]].mean()

    # 计算对应于包围框中心的3D点
    center = box.center()
    x = (center[0] - calib.px()) * box_depth / calib.fx()
    y = (center[1] - calib.py()) * box_depth / calib.fy()
    relative_point = gtsam.Point3(x, y, box_depth)
    quadric_center = pose.transformFrom(relative_point)

    # 使用Lookat计算四次曲面的旋转
    up_vector = pose.transformFrom(gtsam.Point3(0, -1, 0))
    quadric_rotation = gtsam.PinholeCameraCal3_S2.Lookat(
        pose.translation(), quadric_center, up_vector,
        calib).pose().rotation()
    quadric_pose = gtsam.Pose3(quadric_rotation, quadric_center)

    # 从包围框形状计算四次曲面的半径
    tx = (box.xmin() - calib.px()) * box_depth / calib.fx()
    ty = (box.ymin() - calib.py()) * box_depth / calib.fy()
    radii = np.array([np.abs(tx - x), np.abs(ty - y), object_depth])

    # 从位姿和半径构造约束对偶二次曲面
    return gtsam_quadrics.ConstrainedDualQuadric(quadric_pose, radii)


def initialise_quadric_ray_intersection(
        obs_poses: List[gtsam.Pose3], boxes: List[gtsam_quadrics.AlignedBox2],
        state: QuadricSlamState) -> gtsam_quadrics.ConstrainedDualQuadric:
    # 从所有四次曲面的观测中，投射射线到3D空间，并使用它们的最接近的汇聚点来放置四次曲面。
    # 初始方向和大小目前只是简单的猜测。

    # 获取每个观测点的位置
    ps = np.array([op.translation() for op in obs_poses])

    # 获取相机参数
    s = state.system
    assert s.calib_rgb is not None
    calib = s.calib_rgb

    # 获取每个观测点的方向
    # TODO 实际上使用包围框而不是假设中间位置...
    # vs = np.array([op.rotation().matrix()[:, 0] for op in obs_poses])
    vs = []
    for pose, box in zip(obs_poses, boxes):
        center = box.center()
        depth = state.this_step.depth[int(center[1]), int(center[0])]
        x = (center[0] - calib.px()) * depth / calib.fx()
        y = (center[1] - calib.py()) * depth / calib.fy()
        relative_point = gtsam.Point3(x, y, depth)
        direction = pose.rotation().matrix() @ relative_point.vector()
        vs.append(direction / np.linalg.norm(direction))
    vs = np.array(vs)

    # 计算所有射线汇聚点的最近点：
    #   https://stackoverflow.com/a/52089698/1386784
    i_minus_vs = np.eye(3) - (vs[:, :, np.newaxis] @ vs[:, np.newaxis, :])
    quadric_centroid = np.linalg.lstsq(
        i_minus_vs.sum(axis=0),
        (i_minus_vs @ ps[:, :, np.newaxis]).sum(axis=0),
        rcond=None)[0].squeeze()

    # 暂时处理其余部分
    # TODO 改进算法...
    # 从旋转矩阵、平移向量和半径构造约束对偶二次曲面
    return gtsam_quadrics.ConstrainedDualQuadric(
        gtsam.Rot3(), gtsam.Point3(quadric_centroid), [1, 1, 0.1])


def new_factors(current: gtsam.NonlinearFactorGraph,
                previous: gtsam.NonlinearFactorGraph):
    # Figure out the new factors
    fs = (set([current.at(i) for i in range(0, current.size())]) -
          set([previous.at(i) for i in range(0, previous.size())]))

    # Return a NEW graph with the factors
    out = gtsam.NonlinearFactorGraph()
    for f in fs:
        out.add(f)
    return out


def new_values(current: gtsam.Values, previous: gtsam.Values):
    # Figure out new values
    cps, cqs = ps_and_qs_from_values(current)
    pps, pqs = ps_and_qs_from_values(previous)
    vs = {
        **{k: cps[k] for k in list(set(cps.keys()) - set(pps.keys()))},
        **{k: cqs[k] for k in list(set(cqs.keys()) - set(pqs.keys()))}
    }

    # Return NEW values with each of our estimates
    out = gtsam.Values()
    for k, v in vs.items():
        if type(v) == gtsam_quadrics.ConstrainedDualQuadric:
            v.addToValues(out, k)
        else:
            out.insert(k, v)
    return out


def ps_and_qs_from_values(values: gtsam.Values):
    # TODO there's got to be a better way to access the typed values...
    return ({
        k: values.atPose3(k)
        for k in values.keys()
        if gtsam.Symbol(k).string()[0] == 'x'
    }, {
        k: gtsam_quadrics.ConstrainedDualQuadric.getFromValues(values, k)
        for k in values.keys() 
        if gtsam.Symbol(k).string()[0] == 'q'
    })
