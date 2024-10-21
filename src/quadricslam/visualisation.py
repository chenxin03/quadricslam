from distinctipy import get_colors
from matplotlib.patches import Patch
from typing import Dict
import gtsam
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from typing import Union, Tuple, List

from .utils import ps_and_qs_from_values

import pudb

# 初始化图形和轴对象
fig, ax, color_map, available_colors, placeholder_keys = None, None, None, None, None

def _axis_limits(ps, qs):
    xs = ([p.translation()[0] for p in ps] + [q.bounds().xmin() for q in qs] +
          [q.bounds().xmax() for q in qs])
    ys = ([p.translation()[1] for p in ps] + [q.bounds().ymin() for q in qs] +
          [q.bounds().ymax() for q in qs])
    return np.min(xs), np.max(xs), np.min(ys), np.max(ys)


def _scale_factor(ps, qs):
    lims = _axis_limits(ps, qs)
    return np.max([lims[1] - lims[0], lims[3] - lims[2]])


def _set_axes_equal(ax):
    # Matplotlib is really ordinary for 3D plots... here's a hack taken from
    # here to get 'square' in 3D:
    #   https://stackoverflow.com/a/31364297/1386784
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def initialise_visualisation(num: int = 20):
    global fig, ax, color_map, available_colors, placeholder_keys
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.show(block=False)
    # 全局存储颜色映射
    color_map = {}
    available_colors = plt.cm.tab20(np.linspace(0, 1, num))
    placeholder_keys = [str(i) for i in range(num)]


def visualise(values: gtsam.Values = None, 
              labels: Dict[int, str] = None, 
              block: bool = False, 
              num: int = 20
              ):
    global fig, ax, color_map, available_colors, placeholder_keys

    global_params = [fig, ax, color_map, available_colors, placeholder_keys]
    if any(param is None for param in global_params):
        initialise_visualisation(num)

    # 生成颜色映射
    ls = set(labels.values())
    new_labels = ls.difference(color_map.keys())
    if new_labels:
        for label in new_labels:
            if placeholder_keys:
                placeholder = placeholder_keys.pop(0)
                color_map[label] = available_colors[int(placeholder)]
            else:
                raise ValueError("No more available colors. Increase the number of initial colors.")

    # 获取最新的椭圆体位姿估计值
    full_ps, full_qs = ps_and_qs_from_values(values)
    # 计算缩放因子
    sf = 0.1 * _scale_factor(full_ps.values(), full_qs.values())
    # print(sf)
    # sf = 1.0
    # 提取位姿矩阵
    ps = [p.matrix() for p in full_ps.values()]
    # 提取坐标和方向信息
    pxs, pys, pzs, pxus, pxvs, pxws, pyus, pyvs, pyws, pzus, pzvs, pzws = (
        np.array([p[0, 3] for p in ps]),
        np.array([p[1, 3] for p in ps]),
        np.array([p[2, 3] for p in ps]),
        np.array([p[0, 0] for p in ps]),
        np.array([p[1, 0] for p in ps]),
        np.array([p[2, 0] for p in ps]),
        np.array([p[0, 1] for p in ps]),
        np.array([p[1, 1] for p in ps]),
        np.array([p[2, 1] for p in ps]),
        np.array([p[0, 2] for p in ps]),
        np.array([p[1, 2] for p in ps]),
        np.array([p[2, 2] for p in ps]),
    )

    plt.ion()
    # 清除之前的图像
    ax.clear()

    alphas = np.linspace(0.2, 1, len(ps))
    for i in range(1, len(ps)):
        ax.plot(pxs[i - 1:i + 1],
                pys[i - 1:i + 1],
                pzs[i - 1:i + 1],
                color='k',
                alpha=alphas[i])
    ax.quiver(pxs, pys, pzs, pxus * sf, pxvs * sf, pxws * sf, color='r')
    ax.quiver(pxs, pys, pzs, pyus * sf, pyvs * sf, pyws * sf, color='g')
    ax.quiver(pxs, pys, pzs, pzus * sf, pzvs * sf, pzws * sf, color='b')

    for k, q in full_qs.items():
        visualise_ellipsoid(q.pose().matrix(), q.radii(), color_map[labels[k]])

    print(f'绘制{len(full_qs)}个object')
    # Plot a legend for quadric colours
    ax.legend(handles=[
        Patch(facecolor=c, edgecolor=c, label=l) for l, c in color_map.items()
    ])

    # 设置坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.autoscale()
    _set_axes_equal(ax)

    # 显示更新后的图像
    plt.draw()
    plt.pause(0.01)

    if block:
        plt.ioff()
        plt.show(block=True)
    else:
        plt.show()


def visualise_fromlist(cap: List[np.ndarray], 
              Obj: List[List[np.ndarray]]
              ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cap_pose = cap
    obj_pose, obj_radius, labels, colors = Obj 

    # 获取唯一的标签
    unique_labels = list(set(labels))

    rgba = []
    for color in colors:
        rgb = [float(i) / 255.0 for i in color]
        rgba.append(rgb + [1])
    # print(rgba)

    # 缩放系数
    sf = 0.5

    # 提取坐标和方向信息
    pxs, pys, pzs, pxus, pxvs, pxws, pyus, pyvs, pyws, pzus, pzvs, pzws = (
        np.array([p[0, 3] for p in cap_pose]),
        np.array([p[1, 3] for p in cap_pose]),
        np.array([p[2, 3] for p in cap_pose]),
        np.array([p[0, 0] for p in cap_pose]),
        np.array([p[1, 0] for p in cap_pose]),
        np.array([p[2, 0] for p in cap_pose]),
        np.array([p[0, 1] for p in cap_pose]),
        np.array([p[1, 1] for p in cap_pose]),
        np.array([p[2, 1] for p in cap_pose]),
        np.array([p[0, 2] for p in cap_pose]),
        np.array([p[1, 2] for p in cap_pose]),
        np.array([p[2, 2] for p in cap_pose]),
    )

    ax.clear()

    alphas = np.linspace(0.2, 1, len(cap_pose))
    for i in range(1, len(cap_pose)):
        ax.plot(pxs[i - 1:i + 1],
                pys[i - 1:i + 1],
                pzs[i - 1:i + 1],
                color='k',
                alpha=alphas[i])
    ax.quiver(pxs, pys, pzs, pxus * sf, pxvs * sf, pxws * sf, color='r')
    ax.quiver(pxs, pys, pzs, pyus * sf, pyvs * sf, pyws * sf, color='g')
    ax.quiver(pxs, pys, pzs, pzus * sf, pzvs * sf, pzws * sf, color='b')

    for pose, radiu, label, color in zip(obj_pose, obj_radius, labels, rgba):
        visualise_ellipsoid(pose, radiu, color)

    label_counts = {label: labels.count(label) for label in unique_labels}
    # 构建输出字符串
    output = f'绘制{len(labels)}个object：'
    for label in unique_labels:
        output += f' {label_counts[label]} * {label} '
    print(output)

    # 图例
    ax.legend(handles=[
        Patch(facecolor=c, edgecolor=c, label=l) for l, c in zip(labels, rgba)
    ])

    # 设置坐标轴标签
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.autoscale()
    _set_axes_equal(ax)

    # 显示更新后的图像
    plt.draw()
    plt.show(block=True)



def visualise_ellipsoid(pose: np.ndarray, radii: np.ndarray, color):
    # Generate ellipsoid of appropriate size at origin
    SZ = 50
    u, v = np.linspace(0, 2 * np.pi, SZ), np.linspace(0, np.pi, SZ)
    x, y, z = (radii[0] * np.outer(np.cos(u), np.sin(v)),
               radii[1] * np.outer(np.sin(u), np.sin(v)),
               radii[2] * np.outer(np.ones_like(u), np.cos(v)))

    # Rotate the ellipsoid, then translate to centroid
    ps = pose @ np.vstack([
        x.reshape(-1),
        y.reshape(-1),
        z.reshape(-1),
        np.ones(z.reshape(-1).shape)
    ])

    # Plot the ellipsoid
    ax = plt.gca()
    ax.plot_wireframe(
        ps[0, :].reshape(SZ, SZ),
        ps[1, :].reshape(SZ, SZ),
        ps[2, :].reshape(SZ, SZ),
        rstride=4,
        cstride=4,
        edgecolors=color,
        linewidth=0.5,
    )


def read_csv(dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    # 读取CSV文件
    # data_ps = np.genfromtxt(os.path.join(dir, 'CapturePose.csv'), delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
    # data_qs = np.genfromtxt(os.path.join(dir, 'Object.csv'), delimiter=',', skip_header=1, dtype=None, encoding='utf-8')
    data_ps = open(os.path.join(dir, 'CapturePose.csv'))
    data_qs = open(os.path.join(dir, 'Object.csv'))

    # print(data_qs)
    poses_cap = []
    poses_obj = []
    radius = []
    labels = []
    colors = []
    for line in data_ps:
        if line[0] == 'p':
            continue
        data = line.strip().split(',')
        data = [float(i) for i in data]
        # print(data) 
        pose_cap = np.array([[data[0], data[1], data[2], data[3]],
                            [data[4], data[5], data[6], data[7]],
                            [data[8], data[9], data[10], data[11]],
                            [data[12], data[13], data[14], data[15]]])
        poses_cap.append(pose_cap)
    for line in data_qs:
        if line[0] == 'p':
            continue
        data = line.strip().split(',')
        data[0:19] = [float(i) for i in data[0:19]]
        color = [int(color_) for color_ in data[20:]]
        # print(data)
        pose_obj = np.array([[data[0], data[1], data[2], data[3]],
                            [data[4], data[5], data[6], data[7]],
                            [data[8], data[9], data[10], data[11]],
                            [data[12], data[13], data[14], data[15]]])
        poses_obj.append(pose_obj)
        radius.append(np.array([data[16], data[17], data[18]]))
        labels.append(data[19])
        colors.append(color)
    # print(poses_cap)
    # print(poses_obj)
    # print(radius)
    # print(labels)
    # print(colors)
    return poses_cap, [poses_obj, radius, labels, colors]
