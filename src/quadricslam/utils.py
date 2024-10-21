from typing import Callable, List
import gtsam
import gtsam_quadrics
import numpy as np
import time
import os
import cv2

from matplotlib import pyplot as plt
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
    print(f'x:{x}  type:{type(x)}\ny:{y}\ndepth:{box_depth}')
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

    # 从位姿和半径构造二次曲面
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
    calib = [517.3, 516.5, 0, 318.6, 255.3]

    # 获取每个观测点的方向
    # TODO 实际上使用包围框而不是假设中间位置...
    # vs = np.array([op.rotation().matrix()[:, 0] for op in obs_poses])
    vs = []
    for pose, box in zip(obs_poses, boxes):
        xmin = int(box.xmin())
        ymin = int(box.ymin())
        xmax = int(box.xmax())
        ymax = int(box.ymax())
        depth = np.mean(state.this_step.depth[ymin : ymax , xmin : xmax]).astype(np.float32)
        x = float(((xmin + xmax) / 2 - calib[3]) * depth / calib[0])
        y = float(((ymin + ymax) / 2 - calib[4]) * depth / calib[1])
        print(f'x:{x}  type:{type(x)}\ny:{y}\ndepth:{depth}')
        relative_point = np.array([x, y, depth], dtype=float)
        direction = pose.rotation().matrix() @ relative_point
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
        gtsam.Rot3(), np.array(quadric_centroid, dtype=float), [1, 1, 0.1])


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
    ps = {}
    qs = {}

    for k in values.keys():
        symbol = gtsam.Symbol(k).string()[0]
        try:
            if symbol == 'x':
                pose = values.atPose3(k)
                if isinstance(pose, gtsam.Pose3):
                    ps[k] = pose
                else:
                    raise TypeError(f"Expected Pose3, got {type(pose)}")
            elif symbol == 'q':
                quadric = gtsam_quadrics.ConstrainedDualQuadric.getFromValues(values, k)
                if isinstance(quadric, gtsam_quadrics.ConstrainedDualQuadric):
                    qs[k] = quadric
                else:
                    raise TypeError(f"Expected ConstrainedDualQuadric, got {type(quadric)}")
        except Exception as e:
            print(f"Error processing key {k}: {e}")

    return ps, qs


def save_state(state: QuadricSlamState, save_path: str = '/home/cnbot/zz/slam/results/', save_img: bool = False) -> None:
    values = state.system.estimates
    labels = state.system.labels
    ps, qs = ps_and_qs_from_values(values)
    # print(f'ps:{ps}')
    # print(f'qs:{qs}')
    poses_cap = [p.matrix() for p in ps.values()]
    poses_obj = [q.pose().matrix() for _, q in qs.items()]
    radius = [q.radii() for _, q in qs.items()]
    keys = [key for key, _ in qs.items()]
    colors = (plt.cm.tab20(range(len(keys)))[:, :3] * 255).astype(int).tolist()
    # print(colors)
    # colors = {key: tuple(color) for key, color in zip(keys, colors_)}
    timesnap = time.strftime("%m-%d-%H-%M", time.localtime(time.time()))
    base_path = os.path.expanduser(os.path.join(save_path, timesnap))
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
                "radius_x,radius_y,radius_z,label,color_r,color_g,color_b\n")
        i = 0
        for pose_obj, radiu, key in (zip(poses_obj, radius, keys)):
            obj_flat = pose_obj.flatten()
            radiu_flat = radiu.flatten()
            label = str(labels[key])
            color = colors[i]
            color_r = str(color[0])
            color_g = str(color[1]) 
            color_b = str(color[2])
            i += 1
            o.write(",".join(map(str, obj_flat)) + "," + ",".join(map(str, radiu_flat)) + "," + label + ',' + color_r + ',' + color_g + ',' + color_b + "\n")
    print(f'数据已写入：{base_path}')

    if save_img:
        img = state.this_step.rgb
        detections = state.this_step.detections
        i = 0
        for detection in detections:
            img = draw_box(img, detection.bounds, detection.label, colors[i])
            i += 1
        cv2.imwrite(os.path.join(base_path, f'result.png'), img)
        print('检测图已保存')

def draw_box(img, bounds, label, color):
    x_min, y_min, x_max, y_max = map(int, bounds)
    # print(x_min, y_min, x_max, y_max)
    # print(f'color = {color}, shape = {color.shape}, type = {type(color)}')
    # 绘制边界框
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    # 设置标签的字体和颜色
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = x_min
    text_y = y_min - 10 if y_min - 10 > 10 else y_min + 10
    # 绘制标签背景
    cv2.rectangle(img, (text_x, text_y - text_size[1]), 
                (text_x + text_size[0], text_y), color, cv2.FILLED)
    # 绘制标签文字
    cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)
    return img
