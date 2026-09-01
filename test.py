import numpy as np
import trimesh
from scipy.signal import savgol_filter
from scipy import ndimage
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree


# ============================================================
# 参数
# ============================================================

STL_PATH = r"G:\database\db_rib\implant_lib\products\FML02-1801203-AG.stl"

VOXEL_SIZE = 0.5


# ============================================================
# 1. 读取 STL
# ============================================================

mesh = trimesh.load(
    STL_PATH,
    force="mesh"
)

print("=" * 70)
print("单根产品中心线测试")
print("=" * 70)

print()
print("原始 STL")
print("-" * 60)

print("顶点数：", len(mesh.vertices))
print("三角面数：", len(mesh.faces))

print("bounds：")
print(mesh.bounds)

print("extents：")
print(mesh.extents)

print("最大尺寸：")
print(np.max(mesh.extents))

print("最小尺寸：")
print(np.min(mesh.extents))

import numpy as np
import trimesh
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull


# ============================================================
# 产品截面中心线
# ============================================================

def extract_product_centerline(
    mesh,
    n_points=120,
    n_sections=180,
    smooth_window=11,
    smooth_polyorder=3
):
    """
    使用截面中心法提取人工肋骨产品中心线。

    流程：

        STL
          ↓
        PCA确定主方向
          ↓
        沿主方向建立截面
          ↓
        每个截面计算中心
          ↓
        得到原始中心线
          ↓
        平滑
          ↓
        按弧长重采样

    注意：
        这里不进行 voxelization。
        也不进行 skeletonization。
    """

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("mesh 必须是 trimesh.Trimesh")

    if len(mesh.vertices) == 0:
        raise ValueError("mesh 没有顶点")

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64
    )

    print()
    print("=" * 60)
    print("产品截面中心线")
    print("=" * 60)

    # ========================================================
    # 1. PCA
    # ========================================================

    center = vertices.mean(axis=0)

    centered = (
        vertices - center
    )

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    main_axis = eigenvectors[
        :,
        np.argmax(eigenvalues)
    ]

    main_axis = (
        main_axis /
        np.linalg.norm(main_axis)
    )

    print()
    print("PCA主方向：")
    print(main_axis)

    # ========================================================
    # 2. 建立局部坐标系
    # ========================================================

    # 找一个不与主轴平行的参考向量

    ref = np.array(
        [0.0, 0.0, 1.0]
    )

    if abs(
        np.dot(
            ref,
            main_axis
        )
    ) > 0.9:

        ref = np.array(
            [0.0, 1.0, 0.0]
        )

    axis_u = np.cross(
        main_axis,
        ref
    )

    axis_u /= np.linalg.norm(
        axis_u
    )

    axis_v = np.cross(
        main_axis,
        axis_u
    )

    axis_v /= np.linalg.norm(
        axis_v
    )

    # ========================================================
    # 3. 所有顶点投影到主轴
    # ========================================================

    longitudinal = (
        centered @ main_axis
    )

    min_s = np.min(
        longitudinal
    )

    max_s = np.max(
        longitudinal
    )

    print()
    print(
        f"PCA轴向范围："
        f"{min_s:.2f} ~ {max_s:.2f} mm"
    )

    print(
        f"PCA投影长度："
        f"{max_s - min_s:.2f} mm"
    )

    # ========================================================
    # 4. 建立截面
    # ========================================================

    section_positions = np.linspace(
        min_s,
        max_s,
        n_sections
    )

    raw_centers = []

    valid_positions = []

    section_half_width = (
        max_s - min_s
    ) / n_sections * 0.8

    # ========================================================
    # 5. 每个截面计算中心
    # ========================================================

    for s in section_positions:

        mask = (
            np.abs(
                longitudinal - s
            )
            <=
            section_half_width
        )

        section_points = vertices[
            mask
        ]

        if len(section_points) < 3:
            continue

        # ----------------------------------------------------
        # 转换到截面二维坐标
        # ----------------------------------------------------

        relative = (
            section_points
            -
            (
                center
                +
                main_axis * s
            )
        )

        u = relative @ axis_u
        v = relative @ axis_v

        # ----------------------------------------------------
        # 使用二维截面凸包
        # ----------------------------------------------------

        points_2d = np.column_stack(
            [
                u,
                v
            ]
        )

        try:

            hull = ConvexHull(
                points_2d
            )

            hull_points = points_2d[
                hull.vertices
            ]

        except Exception:

            hull_points = points_2d

        # ----------------------------------------------------
        # 截面中心
        #
        # 这里采用截面点的几何中心。
        # ----------------------------------------------------

        section_center_2d = (
            np.mean(
                hull_points,
                axis=0
            )
        )

        # ----------------------------------------------------
        # 转回三维
        # ----------------------------------------------------

        point_3d = (
            center
            +
            main_axis * s
            +
            axis_u *
            section_center_2d[0]
            +
            axis_v *
            section_center_2d[1]
        )

        raw_centers.append(
            point_3d
        )

        valid_positions.append(
            s
        )

    raw_centers = np.asarray(
        raw_centers
    )

    valid_positions = np.asarray(
        valid_positions
    )

    print()
    print(
        f"有效截面数量："
        f"{len(raw_centers)}"
    )

    if len(raw_centers) < 10:

        raise RuntimeError(
            "有效截面太少，无法建立产品中心线。"
        )

    # ========================================================
    # 6. 按主轴方向排序
    # ========================================================

    order = np.argsort(
        valid_positions
    )

    raw_centers = (
        raw_centers[order]
    )

    # ========================================================
    # 7. 平滑
    # ========================================================

    window = min(
        smooth_window,
        len(raw_centers)
    )

    if window % 2 == 0:
        window -= 1

    if window <= smooth_polyorder:
        window = (
            smooth_polyorder + 2
        )

        if window % 2 == 0:
            window += 1

    if len(raw_centers) >= window:

        smoothed = np.column_stack(
            [
                savgol_filter(
                    raw_centers[:, i],
                    window,
                    smooth_polyorder
                )
                for i in range(3)
            ]
        )

    else:

        smoothed = raw_centers.copy()

    # ========================================================
    # 8. 按弧长重采样
    # ========================================================

    segment_lengths = np.linalg.norm(
        np.diff(
            smoothed,
            axis=0
        ),
        axis=1
    )

    cumulative = np.concatenate(
        [
            [0.0],
            np.cumsum(
                segment_lengths
            )
        ]
    )

    total_length = cumulative[-1]

    print()
    print(
        f"截面中心线原始长度："
        f"{total_length:.2f} mm"
    )

    if total_length <= 0:

        raise RuntimeError(
            "中心线长度无效。"
        )

    target_length = np.linspace(
        0,
        total_length,
        n_points
    )

    centerline = np.column_stack(
        [
            np.interp(
                target_length,
                cumulative,
                smoothed[:, i]
            )
            for i in range(3)
        ]
    )

    centerline_smooth = smooth_centerline(
        centerline,
        window=15,
        polyorder=3
    )

    # ========================================================
    # 9. 最终长度
    # ========================================================

    final_length = np.sum(
        np.linalg.norm(
            np.diff(
                centerline_smooth,
                axis=0
            ),
            axis=1
        )
    )

    print()
    print(
        f"产品中心线最终长度："
        f"{final_length:.2f} mm"
    )

    print(
        f"中心线点数："
        f"{len(centerline_smooth)}"
    )

    return centerline_smooth

def smooth_centerline(
        points,
        window=15,
        polyorder=3
):
    """
    对中心线进行 Savitzky-Golay 平滑

    points:
        N×3 numpy array

    window:
        平滑窗口，必须奇数

    polyorder:
        多项式阶数
    """

    points = np.asarray(
        points,
        dtype=np.float64
    )


    if len(points) < window:
        return points


    # -------------------------------------------------
    # 保存端点
    # -------------------------------------------------

    start = points[0].copy()
    end = points[-1].copy()


    # -------------------------------------------------
    # 分别对 XYZ 三个方向平滑
    # -------------------------------------------------

    smooth = np.zeros_like(
        points
    )


    for i in range(3):

        smooth[:, i] = savgol_filter(
            points[:, i],
            window_length=window,
            polyorder=polyorder
        )


    # -------------------------------------------------
    # 恢复端点
    # -------------------------------------------------

    smooth[0] = start
    smooth[-1] = end


    return smooth

mesh = trimesh.load(
    STL_PATH,
    force="mesh"
)


centerline = extract_product_centerline(
    mesh,
    n_points=120
)





line = trimesh.load_path(
    centerline
)

line.export(
    "product_centerline.ply"
)
print()
print("=" * 70)
print("测试结束")
print("=" * 70)