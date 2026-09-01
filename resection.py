

from pathlib import Path

import numpy as np
import trimesh


# ============================================================
# 固定参数
# ============================================================

# ------------------------------------------------------------
# 肿瘤对应肋骨区域的固定距离阈值
#
# 单位：mm
#
# reference_B 的顶点距离 tumor 表面 <= 此值，
# 则认为该位置属于肿瘤对应区域。
# ------------------------------------------------------------

TUMOR_DISTANCE_THRESHOLD = 2.0

TUMOR_EXPANSION = 20.0

# 肿瘤扩张体素精度
TUMOR_VOXEL_SIZE = 1.0

# ============================================================
# 基础检查
# ============================================================

def _check_mesh(
    mesh,
    name
):
    """
    检查输入是否为有效 Trimesh。
    """

    if mesh is None:

        raise ValueError(
            f"{name} 不能为空。"
        )

    if not isinstance(
        mesh,
        trimesh.Trimesh
    ):

        raise TypeError(
            f"{name} 必须是 trimesh.Trimesh。"
        )

    if len(mesh.vertices) == 0:

        raise ValueError(
            f"{name} 没有顶点。"
        )

    if len(mesh.faces) == 0:

        raise ValueError(
            f"{name} 没有三角面。"
        )

    return mesh


# ============================================================
# 网格清理
# ============================================================

def _clean_mesh(
    mesh
):
    """
    对 mesh 做基础清理。

    注意：

    这里不会强制要求 watertight。

    因为 reference_B 本身可能不是封闭实体，
    但只要表面三角网格有效，
    就可以用于最近距离计算。
    """

    mesh = _check_mesh(
        mesh,
        "mesh"
    )

    mesh = mesh.copy()

    # --------------------------------------------------------
    # 删除退化面
    # --------------------------------------------------------

    try:

        mesh.remove_degenerate_faces()

    except Exception:

        pass

    # --------------------------------------------------------
    # 删除重复面
    # --------------------------------------------------------

    try:

        mesh.remove_duplicate_faces()

    except AttributeError:

        try:

            mask = mesh.unique_faces()

            mesh.update_faces(
                mask
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # 删除未引用顶点
    # --------------------------------------------------------

    try:

        mesh.remove_unreferenced_vertices()

    except Exception:

        pass

    # --------------------------------------------------------
    # 修复法向
    # --------------------------------------------------------

    try:

        mesh.fix_normals()

    except Exception:

        pass

    return mesh
def _expand_tumor(
    tumor,
    expansion=TUMOR_EXPANSION
):
    """
    将肿瘤模型向外扩张固定距离。

    方法：

        tumor STL
            ↓
        voxelize
            ↓
        填充实体
            ↓
        3D binary dilation
            ↓
        marching cubes
            ↓
        扩张后的 tumor

    expansion:
        向外扩张距离，单位 mm
    """

    tumor = _check_mesh(
        tumor,
        "tumor"
    )

    if expansion <= 0:
        return tumor.copy()

    print()
    print("-" * 60)
    print("肿瘤模型扩张")
    print("-" * 60)

    print(
        f"原始肿瘤尺寸："
        f"{tumor.bounds}"
    )

    print(
        f"固定扩张距离："
        f"{expansion:.2f} mm"
    )

    print(
        f"体素尺寸："
        f"{TUMOR_VOXEL_SIZE:.2f} mm"
    )

    # ========================================================
    # 1. 肿瘤体素化
    # ========================================================

    voxel = tumor.voxelized(
        pitch=TUMOR_VOXEL_SIZE
    )

    # ========================================================
    # 2. 填充肿瘤内部
    # ========================================================

    try:

        voxel = voxel.fill()

    except Exception as e:

        raise RuntimeError(
            f"肿瘤体素填充失败：{e}"
        ) from e

    matrix = np.asarray(
        voxel.matrix,
        dtype=bool
    )

    print(
        f"原始体素数量："
        f"{np.count_nonzero(matrix):,}"
    )

    # ========================================================
    # 3. 计算膨胀次数
    # ========================================================

    iterations = int(
        np.ceil(
            expansion /
            TUMOR_VOXEL_SIZE
        )
    )

    print(
        f"膨胀次数："
        f"{iterations}"
    )

    # ========================================================
    # 4. 3D 球形结构元素
    # ========================================================

    from scipy.ndimage import (
        binary_dilation
    )

    radius = iterations

    x = np.arange(
        -radius,
        radius + 1
    )

    X, Y, Z = np.meshgrid(
        x,
        x,
        x,
        indexing="ij"
    )

    structure = (
        X * X +
        Y * Y +
        Z * Z
        <=
        radius * radius
    )

    # ========================================================
    # 5. 向外膨胀
    # ========================================================

    from scipy.ndimage import distance_transform_edt

    # ========================================================
    # 距离场膨胀
    # ========================================================

    distance = distance_transform_edt(
        ~matrix
    )

    expanded_matrix = (
            distance
            <=
            expansion / TUMOR_VOXEL_SIZE
    )

    print(
        f"扩张后体素数量："
        f"{np.count_nonzero(expanded_matrix):,}"
    )

    # ========================================================
    # 6. 建立新的 VoxelGrid
    # ========================================================

    # ========================================================
    # 重建 VoxelGrid
    # ========================================================

    # ========================================================
    # 6. 建立新的 VoxelGrid
    # ========================================================

    from trimesh.voxel import VoxelGrid

    expanded_voxel = VoxelGrid(
        encoding=trimesh.voxel.encoding.DenseEncoding(
            expanded_matrix
        ),
        transform=voxel.transform.copy()
    )

    expanded = expanded_voxel.marching_cubes

    expanded.apply_transform(
        voxel.transform
    )

    expanded = _clean_mesh(
        expanded
    )

    print()
    print(
        "扩张后肿瘤尺寸："
    )

    print(
        expanded.bounds
    )

    print(
        f"扩张后顶点："
        f"{len(expanded.vertices):,}"
    )

    print(
        f"扩张后三角面："
        f"{len(expanded.faces):,}"
    )

    print(
        f"扩张后 watertight："
        f"{expanded.is_watertight}"
    )

    return expanded

# ============================================================
# 肿瘤对应区域
# ============================================================

def _tumor_reference_intersection(
    tumor,
    reference
):
    """
    计算肿瘤对应的参考肋骨区域。

    注意：

    函数名称为了兼容原有程序保持不变。

    这里已经不再执行 Boolean intersection。

    实际流程：

        tumor
            ↓
        reference 顶点
            ↓
        计算 reference 顶点到 tumor 表面的最近距离
            ↓
        距离 <= TUMOR_DISTANCE_THRESHOLD
            ↓
        保留对应 reference 三角面

    返回：

        一个从 reference 中提取出来的局部肋骨模型。
    """

    tumor = _check_mesh(
        tumor,
        "tumor"
    )


    reference = _check_mesh(
        reference,
        "reference"
    )

    print()
    print(
        "计算肿瘤与参考肋骨对应区域..."
    )

    print(
        f"固定距离阈值："
        f"{TUMOR_DISTANCE_THRESHOLD:.2f} mm"
    )

    # ========================================================
    # 1. 清理输入
    # ========================================================

    tumor_mesh = _clean_mesh(
        tumor
    )

    reference_mesh = _clean_mesh(
        reference
    )

    # ========================================================
    # 2. 基本信息
    # ========================================================

    print()
    print(
        "肿瘤模型："
    )

    print(
        f"  顶点数："
        f"{len(tumor_mesh.vertices):,}"
    )

    print(
        f"  三角面数："
        f"{len(tumor_mesh.faces):,}"
    )

    print(
        f"  watertight："
        f"{tumor_mesh.is_watertight}"
    )

    print()
    print(
        "参考肋骨模型："
    )

    print(
        f"  顶点数："
        f"{len(reference_mesh.vertices):,}"
    )

    print(
        f"  三角面数："
        f"{len(reference_mesh.faces):,}"
    )

    print(
        f"  watertight："
        f"{reference_mesh.is_watertight}"
    )

    # ========================================================
    # 3. reference 顶点
    # ========================================================

    reference_vertices = np.asarray(
        reference_mesh.vertices,
        dtype=np.float64
    )

    if len(reference_vertices) == 0:

        raise RuntimeError(
            "参考肋骨没有有效顶点。"
        )
    # ========================================================
    # 4. 扩张肿瘤
    # ========================================================

    expanded_tumor = _expand_tumor(
        tumor,
        TUMOR_EXPANSION
    )

    print()
    print(
        "计算参考肋骨顶点与扩张肿瘤重合区域..."
    )

    # ========================================================
    # 5. 点是否在扩张肿瘤内部
    # ========================================================

    reference_vertices = np.asarray(
        reference_mesh.vertices,
        dtype=np.float64
    )

    try:

        vertex_mask = expanded_tumor.contains(
            reference_vertices
        )


    except Exception as e:

        raise RuntimeError(
            f"扩张肿瘤空间判断失败:{e}"
        ) from e
    inside = expanded_tumor.contains(
        reference_vertices
    )

    vertex_mask = inside


    selected_vertex_count = int(
        np.count_nonzero(
            vertex_mask
        )
    )
    print()
    print(
        "肿瘤对应区域筛选："
    )

    print(
        f"参考肋骨总顶点："
        f"{len(reference_vertices):,}"
    )

    print(
        f"扩张区域内顶点："
        f"{selected_vertex_count:,}"
    )

    print(
        f"对应比例："
        f"{selected_vertex_count / len(reference_vertices) * 100:.2f}%"
    )
    # ========================================================
    # 7. 根据 reference 三角面建立区域
    #
    # 只要三角面的三个顶点中有一个进入阈值，
    # 就保留这个面。
    #
    # 这样可以避免：
    #
    #     只选择顶点
    #
    # 导致最终 mesh 出现大量断裂。
    # ========================================================

    faces = np.asarray(
        reference_mesh.faces,
        dtype=np.int64
    )

    face_vertex_mask = vertex_mask[
        faces
    ]

    face_mask = np.any(
        face_vertex_mask,
        axis=1
    )

    selected_faces = faces[
        face_mask
    ]

    print(
        f"对应三角面："
        f"{len(selected_faces):,}"
    )

    if len(selected_faces) == 0:

        raise RuntimeError(
            "找到了肿瘤对应顶点，"
            "但没有找到对应三角面。"
        )

    # ========================================================
    # 8. 提取对应区域 mesh
    # ========================================================

    used_vertices = np.unique(
        selected_faces.reshape(-1)
    )

    new_vertices = reference_vertices[
        used_vertices
    ]

    index_map = -np.ones(
        len(reference_vertices),
        dtype=np.int64
    )

    index_map[
        used_vertices
    ] = np.arange(
        len(used_vertices),
        dtype=np.int64
    )

    new_faces = index_map[
        selected_faces
    ]

    corresponding_region = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )

    corresponding_region = _clean_mesh(
        corresponding_region
    )

    # ========================================================
    # 9. 连通组件检查
    # ========================================================

    try:

        components = (
            corresponding_region.split(
                only_watertight=False
            )
        )

        print()
        print(
            f"肿瘤对应区域连通组件："
            f"{len(components)}"
        )

    except Exception:

        components = []

    # ========================================================
    # 10. 最终检查
    # ========================================================

    if len(
        corresponding_region.vertices
    ) == 0:

        raise RuntimeError(
            "肿瘤对应区域生成失败：没有顶点。"
        )

    if len(
        corresponding_region.faces
    ) == 0:

        raise RuntimeError(
            "肿瘤对应区域生成失败：没有三角面。"
        )

    print()
    print(
        "肿瘤对应肋骨区域计算完成。"
    )

    print(
        f"区域顶点："
        f"{len(corresponding_region.vertices):,}"
    )

    print(
        f"区域三角面："
        f"{len(corresponding_region.faces):,}"
    )

    print(
        f"区域 watertight："
        f"{corresponding_region.is_watertight}"
    )

    return corresponding_region


# ============================================================
# PCA确定肋骨总体方向
# ============================================================

def _get_main_axis(
    mesh
):
    """
    使用 PCA 获得肋骨总体方向。

    注意：

    这里只用于确定肋骨总体方向。

    最终中心线算法仍然由 centerline.py 完成。
    """

    mesh = _check_mesh(
        mesh,
        "mesh"
    )

    points = np.asarray(
        mesh.vertices,
        dtype=np.float64
    )

    center = np.mean(
        points,
        axis=0
    )

    centered = (
        points - center
    )

    covariance = np.cov(
        centered,
        rowvar=False
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    direction = eigenvectors[
        :,
        np.argmax(eigenvalues)
    ]

    norm = np.linalg.norm(
        direction
    )

    if norm < 1e-12:

        raise RuntimeError(
            "无法确定肋骨主方向。"
        )

    direction = (
        direction / norm
    )

    return center, direction


# ============================================================
# 计算患部范围
# ============================================================

def _calculate_affected_range(
    tumor_intersection,
    reference_mesh
):
    """
    根据肿瘤对应肋骨区域确定患部范围。

    返回：

        axis_origin
        axis_direction
        start
        end
        length
    """

    tumor_intersection = _check_mesh(
        tumor_intersection,
        "tumor_intersection"
    )

    reference_mesh = _check_mesh(
        reference_mesh,
        "reference_mesh"
    )

    # --------------------------------------------------------
    # 肋骨总体方向
    # --------------------------------------------------------

    origin, direction = _get_main_axis(
        reference_mesh
    )

    # --------------------------------------------------------
    # 肿瘤对应区域顶点
    # --------------------------------------------------------

    tumor_points = np.asarray(
        tumor_intersection.vertices,
        dtype=np.float64
    )

    projection = (
        tumor_points - origin
    ) @ direction

    start = float(
        np.min(projection)
    )

    end = float(
        np.max(projection)
    )

    length = (
        end - start
    )

    print()
    print(
        "肿瘤对应肋骨范围："
    )

    print(
        f"起点："
        f"{start:.2f} mm"
    )

    print(
        f"终点："
        f"{end:.2f} mm"
    )

    print(
        f"患部长度："
        f"{length:.2f} mm"
    )

    if length <= 0:

        raise RuntimeError(
            "肿瘤对应区域长度无效。"
        )

    return {
        "axis_origin": origin,
        "axis_direction": direction,
        "start": start,
        "end": end,
        "length": length
    }


# ============================================================
# 根据顶点范围裁剪模型
# ============================================================

def _crop_mesh_by_axis(
    mesh,
    axis_origin,
    axis_direction,
    min_position,
    max_position
):
    """
    根据肋骨主轴方向进行空间裁剪。

    采用：

        三角面三个顶点全部位于范围内

    的方式保留三角面。

    这样可以避免错误的面索引。
    """

    mesh = _check_mesh(
        mesh,
        "mesh"
    )

    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64
    )

    faces = np.asarray(
        mesh.faces,
        dtype=np.int64
    )

    projection = (
        vertices - axis_origin
    ) @ axis_direction

    vertex_mask = (
        (projection >= min_position)
        &
        (projection <= max_position)
    )

    face_mask = vertex_mask[
        faces
    ].all(
        axis=1
    )

    selected_faces = faces[
        face_mask
    ]

    if len(selected_faces) == 0:

        return None

    used_vertices = np.unique(
        selected_faces.reshape(-1)
    )

    new_vertices = vertices[
        used_vertices
    ]

    index_map = -np.ones(
        len(vertices),
        dtype=np.int64
    )

    index_map[
        used_vertices
    ] = np.arange(
        len(used_vertices),
        dtype=np.int64
    )

    new_faces = index_map[
        selected_faces
    ]

    result = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )

    result = _clean_mesh(
        result
    )

    return result


# ============================================================
# A1
# ============================================================

def _create_A1(
    patient_A,
    affected_range
):
    """
    A1：

        患者完整肋骨
        -
        肿瘤对应切除区域

    当前 registered_B 实际上是：

        患者 A 配准到 B 坐标系后的模型。

    因此这里直接对传入的模型进行空间裁剪。
    """

    patient_A = _check_mesh(
        patient_A,
        "patient_A"
    )

    origin = affected_range[
        "axis_origin"
    ]

    direction = affected_range[
        "axis_direction"
    ]

    start = affected_range[
        "start"
    ]

    end = affected_range[
        "end"
    ]

    vertices = np.asarray(
        patient_A.vertices,
        dtype=np.float64
    )

    faces = np.asarray(
        patient_A.faces,
        dtype=np.int64
    )

    projection = (
        vertices - origin
    ) @ direction

    # --------------------------------------------------------
    # 保留患部前后两段
    # --------------------------------------------------------

    vertex_mask = (
        (projection < start)
        |
        (projection > end)
    )

    face_mask = vertex_mask[
        faces
    ].all(
        axis=1
    )

    selected_faces = faces[
        face_mask
    ]

    if len(selected_faces) == 0:

        raise RuntimeError(
            "A1生成失败："
            "切除范围覆盖了整个患者肋骨。"
        )

    used_vertices = np.unique(
        selected_faces.reshape(-1)
    )

    new_vertices = vertices[
        used_vertices
    ]

    index_map = -np.ones(
        len(vertices),
        dtype=np.int64
    )

    index_map[
        used_vertices
    ] = np.arange(
        len(used_vertices),
        dtype=np.int64
    )

    new_faces = index_map[
        selected_faces
    ]

    A1 = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        process=False
    )

    A1 = _clean_mesh(
        A1
    )

    print()
    print(
        "A1模拟切除完成。"
    )

    print(
        f"A1顶点数："
        f"{len(A1.vertices):,}"
    )

    print(
        f"A1面数："
        f"{len(A1.faces):,}"
    )

    return A1


# ============================================================
# A2
# ============================================================

def _create_A2(
    reference_B,
    affected_range,
):
    """
    A2：

        B 上对应的患部区域
        +
        两侧 overlap 搭接

    用于：

        1. 中心线提取
        2. 长度筛选
        3. 产品形状匹配
    """

    reference_B = _check_mesh(
        reference_B,
        "reference_B"
    )

    origin = affected_range[
        "axis_origin"
    ]

    direction = affected_range[
        "axis_direction"
    ]

    start = affected_range[
        "start"
    ]

    end = affected_range[
        "end"
    ]

    # --------------------------------------------------------
    # 两侧延伸
    # --------------------------------------------------------

    A2_start = (
        start
    )

    A2_end = (
        end
    )

    print()
    print(
        "生成A2理论重建区域..."
    )

    print(
        f"患部范围："
        f"{start:.2f} ~ {end:.2f} mm"
    )


    A2 = _crop_mesh_by_axis(
        reference_B,
        origin,
        direction,
        A2_start,
        A2_end
    )

    if A2 is None:

        raise RuntimeError(
            "A2生成失败："
            "参考肋骨 B 中没有找到对应区域。"
        )

    return A2


# ============================================================
# 主接口
# ============================================================

def create_reconstruction_region(
    patient_A,
    registered_B,
    tumor
):



    print()
    print(
        "=" * 70
    )

    print(
        "开始计算肿瘤对应区域"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # 1. 输入检查
    # ========================================================

    patient_A = _check_mesh(
        patient_A,
        "patient_A"
    )

    registered_patient = _check_mesh(
        registered_B,
        "registered_B"
    )

    tumor = _check_mesh(
        tumor,
        "tumor"
    )

    # ========================================================
    # 2. 计算肿瘤对应肋骨区域
    #
    # 不再使用 Boolean。
    #
    # 这里得到的是：
    #
    #     reference_B 中距离 tumor <= 2 mm
    #
    # 的对应区域。
    # ========================================================

    tumor_intersection = (
        _tumor_reference_intersection(
            tumor,
            registered_patient
        )
    )

    # ========================================================
    # 3. 根据对应区域计算患部范围
    # ========================================================

    affected_range = (
        _calculate_affected_range(
            tumor_intersection,
            registered_patient
        )
    )

    # ========================================================
    # 4. A1
    #
    # registered_B 是患者 A 配准到 B 坐标系后的模型。
    #
    # 因此 A1 直接在该坐标系中进行切除。
    # ========================================================

    A1 = _create_A1(
        registered_patient,
        affected_range
    )

    # ========================================================
    # 5. A2
    # ========================================================

    A2 = _create_A2(
        registered_patient,
        affected_range
    )

    # ========================================================
    # 6. 输出
    # ========================================================

    print()
    print(
        "=" * 70
    )

    print(
        "A1 / A2 生成完成"
    )

    print(
        "=" * 70
    )

    print()
    print(
        f"A1尺寸范围："
        f"{A1.bounds}"
    )

    print(
        f"A2尺寸范围："
        f"{A2.bounds}"
    )

    return A1, A2


# ============================================================
# 保存 A2
# ============================================================

def save_a2_model(
    A2,
    output_path
):
    """
    保存 A2 模型。

    用于阶段性人工检查。
    """

    A2 = _check_mesh(
        A2,
        "A2"
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    A2.export(
        output_path
    )

    print()
    print(
        "A2模型检查文件已保存："
    )

    print(
        output_path.resolve()
    )

    return output_path