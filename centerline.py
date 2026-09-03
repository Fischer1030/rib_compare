import numpy as np
import trimesh
import networkx as nx
from scipy.interpolate import make_splprep

from scipy.ndimage import binary_fill_holes
from scipy.signal import savgol_filter
from skimage.morphology import skeletonize


# ============================================================
# STL 体素化
# ============================================================

def voxelize_mesh(
        mesh,
        voxel_size=0.5
):

    vox = mesh.voxelized(
        voxel_size
    )

    volume = vox.matrix.copy()

    transform = vox.transform

    return volume, transform
def prepare_volume(volume):
    """
    患者肋骨 Skeletonize 前的体素实体化处理。
    """

    from scipy import ndimage

    print()
    print("-" * 60)
    print("准备 Skeletonize 体素模型")
    print("-" * 60)

    # --------------------------------------------------------
    # 1. 最大体素连通组件
    # --------------------------------------------------------

    labels, num_components = ndimage.label(
        volume,
        structure=ndimage.generate_binary_structure(3, 3)
    )

    if num_components > 1:

        sizes = np.bincount(
            labels.ravel()
        )[1:]

        largest_label = (
            np.argmax(sizes) + 1
        )

        volume = (
            labels == largest_label
        )

        print(
            f"保留最大体素组件："
            f"{int(np.sum(volume)):,} voxels"
        )

    # --------------------------------------------------------
    # 2. 填充内部空腔
    # --------------------------------------------------------

    volume = binary_fill_holes(
        volume
    )

    print(
        f"第一次填充后体素数："
        f"{int(np.sum(volume)):,}"
    )

    return volume

def crop_volume(
        volume,
        padding=3
):
    """
    裁剪体素模型，并返回 crop_origin。
    """

    coords = np.argwhere(volume)

    if len(coords) == 0:
        raise RuntimeError(
            "体素模型为空，无法裁剪。"
        )

    min_coord = coords.min(axis=0)
    max_coord = coords.max(axis=0)

    crop_min = np.maximum(
        min_coord - padding,
        0
    )

    crop_max = np.minimum(
        max_coord + padding + 1,
        volume.shape
    )

    cropped = volume[
        crop_min[0]:crop_max[0],
        crop_min[1]:crop_max[1],
        crop_min[2]:crop_max[2]
    ]

    crop_origin = np.asarray(
        crop_min,
        dtype=np.int64
    )

    print()
    print("-" * 60)
    print("体素裁剪")
    print("-" * 60)

    print(
        f"原始体素尺寸："
        f"{volume.shape}"
    )

    print(
        f"裁剪后体素尺寸："
        f"{cropped.shape}"
    )

    print(
        f"crop_origin："
        f"{crop_origin}"
    )

    return (
        cropped,
        crop_origin
    )
def skeleton_3d(volume):
    """
    对已经完成实体化和裁剪的体素模型进行3D Skeletonize。
    """

    from scipy import ndimage

    print()
    print("-" * 60)
    print("正在进行 3D Skeletonize...")
    print("-" * 60)

    skeleton = skeletonize(volume)

    labels, num_components = ndimage.label(
        skeleton,
        structure=ndimage.generate_binary_structure(3, 3)
    )

    print(
        f"Skeleton体素数量："
        f"{int(np.sum(skeleton)):,}"
    )

    print(
        f"Skeleton 26邻域连通组件："
        f"{num_components}"
    )

    # --------------------------------------------------------
    # 保留最大 Skeleton 连通组件
    # --------------------------------------------------------

    if num_components > 1:

        sizes = np.bincount(
            labels.ravel()
        )[1:]

        largest_label = (
            np.argmax(sizes) + 1
        )

        skeleton = (
            labels == largest_label
        )

        print(
            "警告：Skeleton存在多个连通组件。"
        )

        print(
            f"已保留最大Skeleton组件："
            f"{int(np.sum(skeleton)):,} voxels"
        )

    return skeleton

def path_voxel_to_world(
        node_path,
        coords,
        crop_origin,
        transform
):
    """
    Skeleton longest path:
        cropped voxel coordinates
        →
        original voxel coordinates
        →
        world coordinates
    """

    path_coords = coords[
        node_path
    ].astype(
        np.float64
    )

    # --------------------------------------------------------
    # 恢复 crop 前坐标
    # --------------------------------------------------------

    path_coords += np.asarray(
        crop_origin,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # voxel index:
    # [z, y, x]
    #
    # trimesh:
    # [x, y, z]
    # --------------------------------------------------------

    xyz = path_coords[:, ::1]

    world_coords = trimesh.transform_points(
        xyz,
        transform
    )

    return world_coords
# ============================================================
# 建立 Skeleton 图
# ============================================================
def calculate_arc_length(points):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    if len(points) < 2:
        return np.zeros(
            len(points),
            dtype=np.float64
        )

    segment_lengths = np.linalg.norm(
        np.diff(points, axis=0),
        axis=1
    )

    return np.concatenate([
        [0.0],
        np.cumsum(segment_lengths)
    ])
def _build_skeleton_graph(
        skeleton,
        voxel_size
):

    coords = np.argwhere(
        skeleton
    )

    n = len(coords)

    G = nx.Graph()

    for i in range(n):

        G.add_node(i)

    lookup = {
        tuple(c): i
        for i, c in enumerate(coords)
    }

    # --------------------------------------------------------
    # 26邻域
    # --------------------------------------------------------

    neighbor_offsets = []

    for x in [-1, 0, 1]:

        for y in [-1, 0, 1]:

            for z in [-1, 0, 1]:

                if (
                    x == 0
                    and
                    y == 0
                    and
                    z == 0
                ):
                    continue

                neighbor_offsets.append(
                    np.array(
                        [x, y, z]
                    )
                )

    # --------------------------------------------------------
    # 建图
    # --------------------------------------------------------

    for i, c in enumerate(coords):

        for offset in neighbor_offsets:

            nb = tuple(
                c + offset
            )

            if nb not in lookup:
                continue

            j = lookup[nb]

            if i >= j:
                continue

            distance = (
                np.linalg.norm(
                    offset
                )
                *
                voxel_size
            )

            G.add_edge(
                i,
                j,
                weight=float(distance)
            )

    return G, coords


# ============================================================
# Skeleton 拓扑诊断
# ============================================================

def _diagnose_skeleton_graph(
        G
):

    endpoints = []

    branch_points = []

    for node in G.nodes:

        degree = G.degree(
            node
        )

        if degree == 1:

            endpoints.append(
                node
            )

        elif degree > 2:

            branch_points.append(
                node
            )

    print()
    print("-" * 60)
    print("Skeleton 拓扑诊断")
    print("-" * 60)

    print(
        f"Skeleton节点数："
        f"{G.number_of_nodes()}"
    )

    print(
        f"Skeleton边数："
        f"{G.number_of_edges()}"
    )

    print(
        f"端点数量："
        f"{len(endpoints)}"
    )

    print(
        f"分叉节点数量："
        f"{len(branch_points)}"
    )

    if len(endpoints) == 2:

        print(
            "Skeleton拓扑：标准两端结构。"
        )

    elif len(endpoints) > 2:

        print(
            "警告：Skeleton存在多个端点，"
            "可能存在毛刺或错误骨架分支。"
        )

    elif len(endpoints) == 0:

        print(
            "警告：Skeleton没有端点，"
            "可能存在闭环结构。"
        )

    return (
        endpoints,
        branch_points
    )


# ============================================================
# 根据 PCA 确定两个端部候选节点
# ============================================================

# ============================================================
# 寻找 PCA 两端之间的最佳路径
# ============================================================
def _find_longest_endpoint_path(
        G,
        endpoints
):
    """
    在 Skeleton 图中，
    对所有 Skeleton endpoint 两两寻找路径，
    返回真实空间长度最大的 endpoint-to-endpoint path。
    """

    if len(endpoints) < 2:
        raise RuntimeError(
            "Skeleton端点少于2个，无法提取中心线。"
        )

    print()
    print("-" * 60)
    print("寻找 Skeleton 端点之间的最长路径")
    print("-" * 60)

    best_path = None
    best_length = -np.inf
    best_pair = None

    tested_pairs = 0

    # --------------------------------------------------------
    # endpoint 两两组合
    # --------------------------------------------------------

    for i in range(len(endpoints)):

        start = endpoints[i]

        distances, paths = nx.single_source_dijkstra(
            G,
            start,
            weight="weight"
        )

        for j in range(i + 1, len(endpoints)):

            end = endpoints[j]

            if end not in distances:
                continue

            path_length = distances[end]

            tested_pairs += 1

            if path_length > best_length:

                best_length = path_length

                best_path = paths[end]

                best_pair = (
                    start,
                    end
                )

    if best_path is None:

        raise RuntimeError(
            "无法找到 Skeleton 端点之间的有效路径。"
        )

    print(
        f"测试端点组合："
        f"{tested_pairs}"
    )

    print(
        f"最长路径节点数："
        f"{len(best_path)}"
    )

    print(
        f"最长路径长度："
        f"{best_length:.2f} mm"
    )

    print(
        f"起点节点："
        f"{best_pair[0]}"
    )

    print(
        f"终点节点："
        f"{best_pair[1]}"
    )

    return np.asarray(
        best_path,
        dtype=np.int64
    )

# ============================================================

# 最佳路径
#
# 保持原函数名称和参数完全不变
# ============================================================

def longest_path(
        skeleton,
        voxel_size,
        transform
):

    # ========================================================
    # Skeleton voxel 坐标
    # ========================================================

    coords = np.argwhere(
        skeleton
    )

    n = len(coords)

    print()
    print(
        f"Skeleton节点数量：{n}"
    )

    if n < 5:

        raise RuntimeError(
            "Skeleton节点数量过少，无法提取中心线。"
        )

    # ========================================================
    # 建立 Skeleton 图
    # ========================================================

    G, coords = _build_skeleton_graph(
        skeleton,
        voxel_size
    )

    print(
        "Skeleton图节点：",
        G.number_of_nodes()
    )

    print(
        "Skeleton图边：",
        G.number_of_edges()
    )

    # ========================================================
    # Skeleton 拓扑诊断
    # ========================================================

    endpoints, branch_points = (
        _diagnose_skeleton_graph(
            G
        )
    )


    # ========================================================
    # 找到两端之间的中心路径
    # ========================================================

    node_path = (
        _find_longest_endpoint_path(

            G,
            coords

        )
    )

    # ========================================================
    # voxel → 世界坐标
    # ========================================================

    centerline = []

    for i in node_path:

        p = coords[i]

        xyz = trimesh.transform_points(

            np.array(
                [
                    p[::-1]
                ]
            ),

            transform

        )[0]

        centerline.append(
            xyz
        )

    centerline = np.array(
        centerline
    )

    # ========================================================
    # 计算实际长度
    # ========================================================

    if len(centerline) >= 2:

        segment_lengths = np.linalg.norm(
            np.diff(
                centerline,
                axis=0
            ),
            axis=1
        )

        actual_length = np.sum(
            segment_lengths
        )

    else:

        actual_length = 0

    print()
    print(
        f"最终Skeleton中心线长度："
        f"{actual_length:.2f} mm"
    )

    return centerline


# ============================================================
# 平滑
# ============================================================

def smooth_centerline(
        line,
        window=21
):

    if len(line) < 5:

        return line

    # --------------------------------------------------------
    # window必须为奇数
    # --------------------------------------------------------

    if window % 2 == 0:

        window += 1

    # --------------------------------------------------------
    # 防止 window 大于数据长度
    # --------------------------------------------------------

    if window > len(line):

        window = (
            len(line)
            if len(line) % 2 == 1
            else len(line) - 1
        )

    if window < 5:

        return line

    result = np.zeros_like(
        line
    )

    for i in range(3):

        result[:, i] = (
            savgol_filter(
                line[:, i],
                window,
                3
            )
        )

    return result


# ============================================================
# 重采样
# ============================================================

def resample_centerline(
        line,
        n_points=120
):

    if len(line) < 2:

        return line

    # --------------------------------------------------------
    # 原始弧长
    # --------------------------------------------------------

    length = np.zeros(
        len(line)
    )

    for i in range(
        1,
        len(line)
    ):

        length[i] = (

            length[i - 1]

            +

            np.linalg.norm(
                line[i]
                -
                line[i - 1]
            )

        )

    total_length = (
        length[-1]
    )

    if total_length <= 0:

        return np.repeat(
            line[:1],
            n_points,
            axis=0
        )

    # --------------------------------------------------------
    # 新弧长
    # --------------------------------------------------------

    new_length = np.linspace(

        0,

        total_length,

        n_points

    )

    result = []

    # --------------------------------------------------------
    # 插值
    # --------------------------------------------------------

    for l in new_length:

        idx = np.searchsorted(
            length,
            l
        )

        if idx == 0:

            result.append(
                line[0]
            )

            continue

        if idx >= len(line):

            result.append(
                line[-1]
            )

            continue

        denominator = (
            length[idx]
            -
            length[idx - 1]
        )

        if denominator <= 0:

            result.append(
                line[idx]
            )

            continue

        ratio = (

            l
            -
            length[idx - 1]

        ) / denominator

        p = (

            line[idx - 1]

            +

            ratio
            *
            (
                line[idx]
                -
                line[idx - 1]
            )

        )

        result.append(
            p
        )

    return np.array(
        result
    )


# ============================================================
# 外部调用
#
# 接口保持完全不变
# ============================================================
def extract_centerline(
        mesh,
        voxel_size=0.5,
        n_points=120
):

    print()
    print("=" * 70)
    print("开始提取患者完整肋骨中心线")
    print("=" * 70)

    # ========================================================
    # 1. 体素化
    # ========================================================

    volume, transform = voxelize_mesh(
        mesh,
        voxel_size
    )

    print(
        f"原始体素矩阵："
        f"{volume.shape}"
    )

    print(
        f"原始体素数量："
        f"{int(np.sum(volume)):,}"
    )

    # ========================================================
    # 2. 体素实体化
    # ========================================================

    volume = prepare_volume(
        volume
    )

    # ========================================================
    # 3. 裁剪
    # ========================================================

    cropped_volume, crop_origin = (
        crop_volume(
            volume,
            padding=3
        )
    )

    # ========================================================
    # 4. 3D Skeleton
    # ========================================================

    skeleton = skeleton_3d(
        cropped_volume
    )

    # ========================================================
    # 5. 建立 26 邻域图
    # ========================================================

    G, coords = _build_skeleton_graph(
        skeleton,
        voxel_size
    )

    # ========================================================
    # 6. Skeleton 拓扑诊断
    # ========================================================

    endpoints, branch_points = (
        _diagnose_skeleton_graph(
            G
        )
    )

    if len(endpoints) < 2:

        raise RuntimeError(
            "Skeleton有效端点少于2个。"
        )

    # ========================================================
    # 7. 所有 endpoint 之间寻找最长路径
    # ========================================================

    node_path = (
        _find_longest_endpoint_path(
            G,
            endpoints
        )
    )

    # ========================================================
    # 8. Skeleton path
    #    → 世界坐标
    # ========================================================

    line = path_voxel_to_world(
        node_path,
        coords,
        crop_origin,
        transform
    )

    # ========================================================
    # 9. 原始真实空间弧长
    # ========================================================

    arc_length = calculate_arc_length(
        line
    )

    raw_length = (
        arc_length[-1]
        if len(arc_length) > 0
        else 0.0
    )

    print()
    print(
        f"最长Skeleton中心线长度："
        f"{raw_length:.2f} mm"
    )

    # ========================================================
    # 10. 轻度 SG 平滑
    # ========================================================

    print()
    print(
        "进行轻度 Savitzky-Golay 平滑..."
    )

    line = smooth_centerline(
        line,
        window=9,
    )

    # ========================================================
    # 11. 平滑后重新计算真实弧长
    # ========================================================

    smoothed_arc_length = (
        calculate_arc_length(
            line
        )
    )

    smoothed_length = (
        smoothed_arc_length[-1]
        if len(smoothed_arc_length) > 0
        else 0.0
    )

    print(
        f"平滑后中心线长度："
        f"{smoothed_length:.2f} mm"
    )

    # ========================================================
    # 12. 按真实弧长重新采样
    # ========================================================

    line = resample_centerline(
        line,
        n_points
    )

    # ========================================================
    # 13. 最终长度
    # ========================================================

    final_arc_length = (
        calculate_arc_length(
            line
        )
    )

    final_length = (
        final_arc_length[-1]
        if len(final_arc_length) > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("患者完整肋骨中心线生成完成")
    print("=" * 70)

    print(
        f"最终中心线长度："
        f"{final_length:.2f} mm"
    )

    print(
        f"最终中心线点数："
        f"{len(line)}"
    )

    return line
# ============================================================
# 中心线长度
# ============================================================

def centerline_length(points):

    points = np.asarray(
        points,
        dtype=np.float64
    )

    if len(points) < 2:
        return 0.0

    return float(
        np.sum(
            np.linalg.norm(
                np.diff(points, axis=0),
                axis=1
            )
        )
    )
def extract_subregion_centerline(
        full_centerline,
        subregion_mesh,
        n_points=120,
        distance_threshold=8.0,
        smooth_sigma=2.0,
        min_segment_points=8
):
    """
    从完整肋骨中心线中截取 A2 对应的中心线。

    方法：
        完整中心线
            ↓
        计算每个中心线点到 A2 表面的最近距离
            ↓
        对距离曲线进行平滑
            ↓
        找到连续的低距离区域
            ↓
        截取对应中心线
            ↓
        弧长重采样

    注意：
        不要求 A2 watertight。
        不对 A2 skeletonize。
    """

    import numpy as np
    import trimesh
    from scipy.ndimage import gaussian_filter1d

    # ========================================================
    # 0. 输入检查
    # ========================================================

    full_centerline = np.asarray(
        full_centerline,
        dtype=np.float64
    )

    if len(full_centerline) < 2:
        raise ValueError(
            "完整中心线点数不足。"
        )

    if not isinstance(
        subregion_mesh,
        trimesh.Trimesh
    ):
        raise TypeError(
            "subregion_mesh 必须是 trimesh.Trimesh"
        )

    print()
    print("=" * 70)
    print("从完整中心线截取 A2 中心线")
    print("=" * 70)

    print(
        f"完整中心线点数：{len(full_centerline)}"
    )

    print(
        f"完整中心线长度："
        f"{centerline_length(full_centerline):.2f} mm"
    )

    print()
    print("A2 bounds：")
    print(subregion_mesh.bounds)

    print(
        f"A2 watertight："
        f"{subregion_mesh.is_watertight}"
    )

    # ========================================================
    # 1. 检查 A2
    # ========================================================

    if len(subregion_mesh.vertices) == 0:
        raise RuntimeError(
            "A2 没有有效顶点。"
        )

    # ========================================================
    # 2. 完整中心线 → A2 网格最近距离
    #
    # 不使用 contains()
    # 不要求 watertight
    # ========================================================

    print()
    print(
        "正在计算完整中心线 → A2 表面距离..."
    )

    try:

        closest_points, distances, triangle_id = (
            trimesh.proximity.closest_point(
                subregion_mesh,
                full_centerline
            )
        )

    except Exception as e:

        raise RuntimeError(
            "计算中心线到 A2 表面距离失败。\n"
            f"错误：{e}"
        )

    distances = np.asarray(
        distances,
        dtype=np.float64
    )

    print()
    print(
        "完整中心线 → A2 距离统计："
    )

    print(
        f"最小距离："
        f"{np.min(distances):.3f} mm"
    )

    print(
        f"最大距离："
        f"{np.max(distances):.3f} mm"
    )

    print(
        f"平均距离："
        f"{np.mean(distances):.3f} mm"
    )

    print(
        f"中位数："
        f"{np.median(distances):.3f} mm"
    )

    # ========================================================
    # 3. 输出距离曲线的关键数据
    # ========================================================

    print()
    print(
        "中心线距离序列："
    )

    for i in range(
        0,
        len(distances),
        max(1, len(distances) // 20)
    ):

        print(
            f"index={i:3d}  "
            f"distance={distances[i]:8.3f} mm"
        )

    # ========================================================
    # 4. 平滑距离曲线
    # ========================================================

    smooth_distances = gaussian_filter1d(
        distances,
        sigma=smooth_sigma
    )

    # ========================================================
    # 5. 根据阈值寻找 A2 对应区域
    # ========================================================

    close_mask = (
        smooth_distances
        <=
        distance_threshold
    )

    print()
    print(
        f"距离阈值："
        f"{distance_threshold:.2f} mm"
    )

    print(
        f"满足距离阈值的中心线点："
        f"{np.count_nonzero(close_mask)}"
    )

    # ========================================================
    # 6. 寻找连续区段
    # ========================================================

    segments = []

    start = None

    for i, flag in enumerate(close_mask):

        if flag and start is None:

            start = i

        elif (
            not flag
            and start is not None
        ):

            end = i - 1

            if (
                end - start + 1
                >= min_segment_points
            ):

                segments.append(
                    (start, end)
                )

            start = None

    # 处理最后一个区段
    if start is not None:

        end = len(close_mask) - 1

        if (
            end - start + 1
            >= min_segment_points
        ):

            segments.append(
                (start, end)
            )

    # ========================================================
    # 7. 输出所有候选区段
    # ========================================================

    print()
    print(
        "检测到的连续候选区段："
    )

    if len(segments) == 0:

        raise RuntimeError(
            "\n"
            "没有找到连续的 A2 对应中心线区域。\n"
            "\n"
            f"当前距离阈值："
            f"{distance_threshold:.2f} mm\n"
            f"最小距离："
            f"{np.min(distances):.3f} mm\n"
            "\n"
            "建议尝试提高 distance_threshold。"
        )

    for start, end in segments:

        segment_length = centerline_length(
            full_centerline[
                start:end + 1
            ]
        )

        print(
            f"  {start:3d} ~ {end:3d}   "
            f"点数={end - start + 1:3d}   "
            f"长度={segment_length:.2f} mm"
        )

    # ========================================================
    # 8. 选择最佳区段
    #
    # 不单纯选择最长区段。
    #
    # 使用：
    #   区段长度
    #   +
    #   区段内部平均距离
    #
    # 综合判断。
    # ========================================================

    best_segment = None
    best_score = -np.inf

    for start, end in segments:

        segment_distances = smooth_distances[
            start:end + 1
        ]

        segment_length = centerline_length(
            full_centerline[
                start:end + 1
            ]
        )

        mean_distance = np.mean(
            segment_distances
        )

        # 距离越小越好
        distance_score = (
            1.0 /
            (
                mean_distance
                + 1e-6
            )
        )

        # 长度适当加权
        length_score = np.sqrt(
            max(segment_length, 1.0)
        )

        score = (
            distance_score
            *
            length_score
        )

        if score > best_score:

            best_score = score
            best_segment = (
                start,
                end
            )

    best_start, best_end = best_segment

    print()
    print(
        "选择的 A2 主体中心线区域："
    )

    print(
        f"索引："
        f"{best_start} ~ {best_end}"
    )

    print(
        f"点数："
        f"{best_end - best_start + 1}"
    )

    print(
        f"区域长度："
        f"{centerline_length(full_centerline[best_start:best_end + 1]):.2f} mm"
    )

    print(
        f"区域平均距离："
        f"{np.mean(smooth_distances[best_start:best_end + 1]):.3f} mm"
    )

    # ========================================================
    # 9. 边界缓冲
    # ========================================================

    index_margin = 2

    start_index = max(
        0,
        best_start - index_margin
    )

    end_index = min(
        len(full_centerline) - 1,
        best_end + index_margin
    )

    selected = full_centerline[
        start_index:
        end_index + 1
    ]

    print()
    print(
        f"最终截取索引："
        f"{start_index} ~ {end_index}"
    )

    print(
        f"截取点数："
        f"{len(selected)}"
    )

    # ========================================================
    # 10. 弧长重采样
    # ========================================================

    segment_lengths = np.linalg.norm(
        np.diff(
            selected,
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

    if total_length <= 0:

        raise RuntimeError(
            "截取后的 A2 中心线长度无效。"
        )

    target = np.linspace(
        0.0,
        total_length,
        n_points
    )

    centerline = np.column_stack(
        [
            np.interp(
                target,
                cumulative,
                selected[:, axis]
            )
            for axis in range(3)
        ]
    )

    # ========================================================
    # 11. 输出结果
    # ========================================================

    print()
    print("=" * 70)
    print("A2中心线生成完成")
    print("=" * 70)

    print(
        f"A2中心线长度："
        f"{centerline_length(centerline):.2f} mm"
    )

    print(
        f"中心线点数："
        f"{len(centerline)}"
    )

    print()
    print("A2中心线端点：")

    print(
        "起点：",
        centerline[0]
    )

    print(
        "终点：",
        centerline[-1]
    )

    return centerline

def fit_and_visualize_centerline(
        centerline,
        output_path,
        label="A2",
        smoothing=0.0,
        n_fit_points=300
):
    """
    对离散中心线进行 B-spline 拟合，并将：
    1. 原始中心线
    2. B-spline 拟合曲线

    绘制在同一张图中，用于验证拟合效果。

    Parameters
    ----------
    centerline : (N, 3) ndarray
        原始中心线点
    output_path : str or Path
        输出图片路径
    label : str
        中心线名称
    smoothing : float
        B-spline 平滑参数。
        0 表示尽可能通过原始点。
        数值越大，曲线越平滑。
    n_fit_points : int
        拟合曲线上重新采样的点数。
    """

    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import splprep, splev

    centerline = np.asarray(
        centerline,
        dtype=np.float64
    )

    if centerline.ndim != 2 or centerline.shape[1] != 3:
        raise ValueError(
            f"centerline 应为 (N, 3)，实际为 {centerline.shape}"
        )

    if len(centerline) < 4:
        raise ValueError(
            "中心线点数太少，无法进行 B-spline 拟合"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 1. 按中心线弧长建立参数 t
    # ------------------------------------------------------------

    segment_lengths = np.linalg.norm(
        np.diff(centerline, axis=0),
        axis=1
    )

    cumulative = np.concatenate([
        [0.0],
        np.cumsum(segment_lengths)
    ])

    total_length = cumulative[-1]

    if total_length <= 0:
        raise ValueError("中心线长度为 0")

    t = cumulative / total_length

    # ------------------------------------------------------------
    # 2. B-spline 拟合
    # ------------------------------------------------------------

    spline_params, u = make_splprep(
        centerline.T,
        u=t,
        s=smoothing,
        k=3
    )

    # ------------------------------------------------------------
    # 3. 在拟合曲线上重新采样
    # ------------------------------------------------------------

    u_fit = np.linspace(
        0.0,
        1.0,
        n_fit_points
    )

    fitted = spline_params(u_fit).T

    # ------------------------------------------------------------
    # 4. 计算拟合曲线长度
    # ------------------------------------------------------------

    fitted_length = np.sum(
        np.linalg.norm(
            np.diff(fitted, axis=0),
            axis=1
        )
    )

    # ------------------------------------------------------------
    # 5. 计算原始点到拟合曲线的大致误差
    # ------------------------------------------------------------

    from scipy.spatial import cKDTree

    tree = cKDTree(fitted)

    distances, _ = tree.query(
        centerline
    )

    rmse = np.sqrt(
        np.mean(distances ** 2)
    )

    max_error = np.max(distances)

    # ------------------------------------------------------------
    # 6. 绘图
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # 原始中心线
    ax.plot(
        centerline[:, 0],
        centerline[:, 1],
        centerline[:, 2],
        "o",
        markersize=3,
        alpha=0.5,
        label=f"{label} Raw"
    )

    # B-spline
    ax.plot(
        fitted[:, 0],
        fitted[:, 1],
        fitted[:, 2],
        linewidth=3,
        label=f"{label} B-spline"
    )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    ax.set_title(
        f"{label} Centerline B-spline Fit\n"
        f"Raw Length = {total_length:.2f} mm | "
        f"Fit Length = {fitted_length:.2f} mm | "
        f"RMSE = {rmse:.3f} mm"
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)

    print()
    print("=" * 70)
    print(f"{label} B-spline 拟合结果")
    print("=" * 70)
    print(f"原始点数       ：{len(centerline)}")
    print(f"拟合点数       ：{len(fitted)}")
    print(f"原始中心线长度 ：{total_length:.2f} mm")
    print(f"拟合曲线长度   ：{fitted_length:.2f} mm")
    print(f"拟合 RMSE      ：{rmse:.3f} mm")
    print(f"最大误差       ：{max_error:.3f} mm")
    print(f"可视化         ：{output_path}")
    print("=" * 70)

    return fitted, spline_params