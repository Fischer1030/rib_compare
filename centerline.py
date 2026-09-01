import numpy as np
import trimesh
import networkx as nx

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


# ============================================================
# 3D Skeleton
# ============================================================

def skeleton_3d(
        volume
):

    volume = binary_fill_holes(
        volume
    )

    from scipy import ndimage

    print()
    print("-" * 60)
    print("Skeletonize 前体素连通性检查")
    print("-" * 60)

    voxel_labels, voxel_components = ndimage.label(
        volume,
        structure=ndimage.generate_binary_structure(3, 3)
    )

    component_sizes = np.bincount(
        voxel_labels.ravel()
    )[1:]

    component_sizes = np.sort(
        component_sizes
    )[::-1]

    print(
        f"体素总数：{int(np.sum(volume)):,}"
    )

    print(
        f"体素连通组件：{voxel_components}"
    )

    print(
        "最大组件体素数：",
        component_sizes[0]
        if len(component_sizes) > 0
        else 0
    )

    print(
        "前10个组件：",
        component_sizes[:10]
    )

    # --------------------------------------------------------
    # 如果体素模型存在多个组件
    # 保留最大组件
    # --------------------------------------------------------

    if voxel_components > 1:

        largest_label = (
            np.argmax(
                component_sizes
            ) + 1
        )

        volume = (
            voxel_labels == largest_label
        )

        print(
            f"Skeletonize前已保留最大体素组件："
            f"{int(np.sum(volume)):,} voxels"
        )

    # --------------------------------------------------------
    # Skeletonize
    # --------------------------------------------------------

    print()
    print("正在进行 3D Skeletonize...")

    skeleton = skeletonize(
        volume
    )

    # --------------------------------------------------------
    # Skeleton 连通性
    # --------------------------------------------------------

    structure = ndimage.generate_binary_structure(
        3,
        3
    )

    labels, num_components = ndimage.label(
        skeleton,
        structure=structure
    )

    print()
    print("-" * 60)
    print("Skeleton 连通性检查")
    print("-" * 60)

    print(
        f"Skeleton体素数量："
        f"{int(np.sum(skeleton)):,}"
    )

    print(
        f"Skeleton 26邻域连通组件："
        f"{num_components}"
    )

    # --------------------------------------------------------
    # 如果 Skeleton 出现多个组件
    # 保留最大 Skeleton
    # --------------------------------------------------------

    if num_components > 1:

        sizes = np.bincount(
            labels.ravel()
        )[1:]

        largest_label = (
            np.argmax(
                sizes
            ) + 1
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


# ============================================================
# Skeleton 坐标
# ============================================================

def skeleton_points(
        skeleton,
        transform
):

    idx = np.argwhere(
        skeleton
    )

    points = []

    for p in idx:

        xyz = trimesh.transform_points(

            np.array(
                [
                    p[::-1]
                ]
            ),

            transform

        )[0]

        points.append(
            xyz
        )

    return np.array(
        points
    )


# ============================================================
# 建立 Skeleton 图
# ============================================================

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
# PCA 主轴
# ============================================================

def _calculate_pca_axis(
        coords
):

    if len(coords) < 3:

        raise RuntimeError(
            "Skeleton节点数量不足，无法进行PCA。"
        )

    center = np.mean(
        coords,
        axis=0
    )

    centered = (
        coords
        -
        center
    )

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvalues = eigenvalues[
        order
    ]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    axis = eigenvectors[:, 0]

    axis = axis / np.linalg.norm(
        axis
    )

    return (
        center,
        axis,
        eigenvalues
    )


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

def _find_end_candidates(
        G,
        coords,
        axis,
        center
):

    # --------------------------------------------------------
    # 将所有 Skeleton 节点投影到 PCA 主轴
    # --------------------------------------------------------

    centered = (
        coords
        -
        center
    )

    projection = (
        centered
        @
        axis
    )

    min_projection = np.min(
        projection
    )

    max_projection = np.max(
        projection
    )

    total_range = (
        max_projection
        -
        min_projection
    )

    if total_range <= 0:

        raise RuntimeError(
            "Skeleton PCA主轴长度为0。"
        )

    # --------------------------------------------------------
    # 端部区域
    #
    # 默认取主轴两端 10%
    # --------------------------------------------------------

    end_fraction = 0.10

    low_limit = (
        min_projection
        +
        total_range
        *
        end_fraction
    )

    high_limit = (
        max_projection
        -
        total_range
        *
        end_fraction
    )

    low_candidates = np.where(
        projection <= low_limit
    )[0]

    high_candidates = np.where(
        projection >= high_limit
    )[0]

    # --------------------------------------------------------
    # 如果候选太少
    # 放宽到15%
    # --------------------------------------------------------

    if len(low_candidates) == 0:

        low_limit = (
            min_projection
            +
            total_range
            *
            0.15
        )

        low_candidates = np.where(
            projection <= low_limit
        )[0]

    if len(high_candidates) == 0:

        high_limit = (
            max_projection
            -
            total_range
            *
            0.15
        )

        high_candidates = np.where(
            projection >= high_limit
        )[0]

    # --------------------------------------------------------
    # 优先选择真正的 Skeleton endpoint
    # --------------------------------------------------------

    endpoints = [
        node
        for node in G.nodes
        if G.degree(node) == 1
    ]

    low_endpoints = [
        node
        for node in endpoints
        if projection[node] <= low_limit
    ]

    high_endpoints = [
        node
        for node in endpoints
        if projection[node] >= high_limit
    ]

    if len(low_endpoints) > 0:

        low_candidates = np.array(
            low_endpoints,
            dtype=int
        )

    if len(high_endpoints) > 0:

        high_candidates = np.array(
            high_endpoints,
            dtype=int
        )

    print()
    print("-" * 60)
    print("PCA端部候选节点")
    print("-" * 60)

    print(
        f"PCA主轴范围："
        f"{min_projection:.2f} ~ "
        f"{max_projection:.2f} voxel"
    )

    print(
        f"左端候选节点："
        f"{len(low_candidates)}"
    )

    print(
        f"右端候选节点："
        f"{len(high_candidates)}"
    )

    return (
        low_candidates,
        high_candidates,
        projection
    )


# ============================================================
# 寻找 PCA 两端之间的最佳路径
# ============================================================

def _find_endpoint_to_endpoint_path(
        G,
        coords,
        low_candidates,
        high_candidates,
        projection,
        voxel_size
):

    if (
        len(low_candidates) == 0
        or
        len(high_candidates) == 0
    ):

        raise RuntimeError(
            "无法找到A2 Skeleton的两个端部候选区域。"
        )

    # --------------------------------------------------------
    # 为了避免候选节点过多
    # 分别选距离两端投影极值最近的若干节点
    # --------------------------------------------------------

    low_target = np.min(
        projection
    )

    high_target = np.max(
        projection
    )

    low_candidates = sorted(
        low_candidates,
        key=lambda i:
        abs(
            projection[i]
            -
            low_target
        )
    )[:20]

    high_candidates = sorted(
        high_candidates,
        key=lambda i:
        abs(
            projection[i]
            -
            high_target
        )
    )[:20]

    print()
    print("-" * 60)
    print("寻找 PCA 两端之间的中心路径")
    print("-" * 60)

    best_path = None

    best_length = np.inf

    best_pair = None

    tested_pairs = 0

    # --------------------------------------------------------
    # 对候选端点进行 Dijkstra
    #
    # 注意：
    # 这里使用“最短路径”，而不是全图最长路径。
    #
    # 目的：
    # 避免 Skeleton 中存在回环时，
    # 路径绕圈导致长度严重异常。
    # --------------------------------------------------------

    for start in low_candidates:

        try:

            distances, paths = (
                nx.single_source_dijkstra(
                    G,
                    start,
                    weight="weight"
                )
            )

        except Exception:

            continue

        for end in high_candidates:

            if end not in distances:
                continue

            path = paths[end]

            path_length = (
                distances[end]
            )

            tested_pairs += 1

            if path_length < best_length:

                best_length = (
                    path_length
                )

                best_path = path

                best_pair = (
                    start,
                    end
                )

    if best_path is None:

        raise RuntimeError(
            "无法建立A2两端之间的Skeleton路径。"
        )

    print(
        f"测试端点组合："
        f"{tested_pairs}"
    )

    print(
        f"最终路径节点："
        f"{len(best_path)}"
    )

    print(
        f"Skeleton端到端路径长度："
        f"{best_length:.2f} mm"
    )

    print(
        f"起点投影："
        f"{projection[best_pair[0]]:.2f}"
    )

    print(
        f"终点投影："
        f"{projection[best_pair[1]]:.2f}"
    )

    return np.array(
        best_path,
        dtype=int
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
    # PCA
    # ========================================================

    center, axis, eigenvalues = (
        _calculate_pca_axis(
            coords
        )
    )

    print()
    print("-" * 60)
    print("Skeleton PCA 主轴")
    print("-" * 60)

    print(
        f"PCA中心："
        f"{center}"
    )

    print(
        f"PCA主轴："
        f"{axis}"
    )

    print(
        f"PCA特征值："
        f"{eigenvalues}"
    )

    if (
        eigenvalues[0]
        >
        0
    ):

        linearity = (
            eigenvalues[0]
            /
            (
                eigenvalues[1]
                +
                eigenvalues[2]
                +
                1e-12
            )
        )

    else:

        linearity = 0

    print(
        f"PCA主轴线性度："
        f"{linearity:.3f}"
    )

    # ========================================================
    # 确定两端候选区域
    # ========================================================

    (
        low_candidates,
        high_candidates,
        projection
    ) = _find_end_candidates(

        G,
        coords,
        axis,
        center

    )

    # ========================================================
    # 找到两端之间的中心路径
    # ========================================================

    node_path = (
        _find_endpoint_to_endpoint_path(

            G,
            coords,

            low_candidates,
            high_candidates,

            projection,

            voxel_size

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
        n_points=300
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
        n_points=300
):

    print()
    print("=" * 70)
    print("开始提取中心线")
    print("=" * 70)

    # ========================================================
    # 体素化
    # ========================================================

    volume, transform = (
        voxelize_mesh(

            mesh,

            voxel_size

        )
    )

    print(
        f"体素矩阵："
        f"{volume.shape}"
    )

    print(
        f"实心体素数量："
        f"{int(np.sum(volume)):,}"
    )

    # ========================================================
    # Skeleton
    # ========================================================

    skeleton = (
        skeleton_3d(
            volume
        )
    )

    # ========================================================
    # Skeleton节点
    # ========================================================

    pts = (
        skeleton_points(

            skeleton,

            transform

        )
    )

    print(
        f"Skeleton世界坐标点数："
        f"{len(pts)}"
    )

    # ========================================================
    # 核心：
    # PCA + 两端约束 + Skeleton路径
    # ========================================================

    line = (
        longest_path(

            skeleton,

            voxel_size,

            transform

        )
    )

    # ========================================================
    # 平滑
    # ========================================================

    print()
    print(
        "进行中心线轻度平滑..."
    )

    line = (
        smooth_centerline(
            line
        )
    )

    # ========================================================
    # 重采样
    # ========================================================

    line = (
        resample_centerline(

            line,

            n_points

        )
    )

    # ========================================================
    # 最终长度
    # ========================================================

    if len(line) >= 2:

        final_length = np.sum(
            np.linalg.norm(
                np.diff(
                    line,
                    axis=0
                ),
                axis=1
            )
        )

    else:

        final_length = 0

    print()
    print(
        f"中心线最终长度："
        f"{final_length:.2f} mm"
    )

    print(
        f"中心线点数："
        f"{len(line)}"
    )

    return line