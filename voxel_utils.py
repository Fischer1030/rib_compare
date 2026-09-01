import numpy as np

from scipy import ndimage

import trimesh


# ============================================================
# STL → 体素
# ============================================================

def voxelize_mesh(
    mesh,
    voxel_size
):

    print()
    print("=" * 60)
    print("STL体素化")
    print("=" * 60)

    print(
        f"体素尺寸：{voxel_size:.3f} mm"
    )

    voxelized = mesh.voxelized(
        pitch=voxel_size
    )

    print(
        f"表面体素数量："
        f"{len(voxelized.points):,}"
    )

    print(
        "正在填充模型内部..."
    )

    try:

        filled = voxelized.fill()

    except Exception:

        filled = voxelized

    matrix = filled.matrix.astype(
        bool
    )

    transform = filled.transform

    print(
        f"体素矩阵：{matrix.shape}"
    )

    print(
        f"实心体素数量："
        f"{matrix.sum():,}"
    )



    return matrix, transform


# ============================================================
# 删除小组件
# ============================================================

def remove_small_components(
    volume,
    min_voxels
):

    print()
    print(
        "=" * 60
    )

    print(
        "体素级孤立组件清理"
    )

    structure = np.ones(
        (3, 3, 3),
        dtype=bool
    )

    labels, count = ndimage.label(
        volume,
        structure=structure
    )

    print(
        f"连通区域数量：{count}"
    )

    if count <= 1:

        print(
            "没有需要删除的小组件。"
        )

        return volume

    sizes = np.bincount(
        labels.ravel()
    )

    keep = sizes >= min_voxels

    keep[0] = False

    cleaned = keep[
        labels
    ]

    removed = (
        volume.sum()
        - cleaned.sum()
    )

    print(
        f"删除体素数量：{removed:,}"
    )

    return cleaned


# ============================================================
# 填充内部空腔
# ============================================================

def fill_internal_cavities(
    volume
):

    before = int(
        volume.sum()
    )

    filled = ndimage.binary_fill_holes(
        volume
    )

    after = int(
        filled.sum()
    )

    print()
    print(
        f"填充前体素：{before:,}"
    )

    print(
        f"填充后体素：{after:,}"
    )

    print(
        f"新增体素：{after - before:,}"
    )

    return filled


# ============================================================
# 实体化
# ============================================================

def solidify_volume(
    volume,
    iterations,
    voxel_size
):

    print()
    print(
        "=" * 60
    )

    print(
        "实体化肋骨"
    )

    print(
        f"闭运算迭代次数：{iterations}"
    )

    if iterations <= 0:

        return volume

    structure = np.ones(
        (3, 3, 3),
        dtype=bool
    )

    result = ndimage.binary_closing(
        volume,
        structure=structure,
        iterations=iterations
    )

    return result


# ============================================================
# 体素诊断
# ============================================================

def diagnose_volume(
    volume,
    voxel_size,
    title=""
):

    print()
    print("=" * 60)

    if title:
        print(title)

    print("=" * 60)

    count = int(
        volume.sum()
    )

    total = volume.size

    ratio = (
        count / total * 100
        if total > 0
        else 0
    )

    print(
        f"体素矩阵：{volume.shape}"
    )

    print(
        f"实心体素数量：{count:,}"
    )

    print(
        f"体素占比：{ratio:.4f}%"
    )

    coords = np.argwhere(volume)

    if len(coords) == 0:

        return

    minimum = coords.min(
        axis=0
    )

    maximum = coords.max(
        axis=0
    )

    size = (
        maximum - minimum + 1
    ) * voxel_size

    print(
        f"X方向长度：{size[0]:.2f} mm"
    )

    print(
        f"Y方向长度：{size[1]:.2f} mm"
    )

    print(
        f"Z方向长度：{size[2]:.2f} mm"
    )