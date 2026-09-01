from pathlib import Path

import numpy as np
import trimesh
from mesh_utils import clean_mesh

# ============================================================
# 合并多个模型
# ============================================================

def merge_meshes(
    meshes
):

    valid_meshes = []

    for mesh in meshes:

        if mesh is None:
            continue

        if len(mesh.vertices) == 0:
            continue

        valid_meshes.append(
            mesh.copy()
        )

    if not valid_meshes:

        raise ValueError(
            "没有可用于重建的STL模型。"
        )

    return trimesh.util.concatenate(
        valid_meshes
    )


# ============================================================
# 基础清理
# ============================================================

def clean_reconstruction(mesh):

    print()
    print("-" * 60)
    print("清理最终重建模型")
    print("-" * 60)

    mesh = clean_mesh(
        mesh
    )

    return mesh

# ============================================================
# F + A1 + P
# ============================================================

def build_reconstruction(
    full_chest,
    patient_A1,
    final_product
):

    reconstruction = merge_meshes([
        full_chest,
        patient_A1,
        final_product
    ])

    reconstruction = clean_reconstruction(
        reconstruction
    )

    return reconstruction


# ============================================================
# 保存最终模型
# ============================================================

def save_reconstruction(
    mesh,
    output_path
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    mesh.export(
        output_path
    )

    print()
    print("=" * 70)
    print("最终重建模型")
    print("=" * 70)

    print(
        output_path.resolve()
    )

    return output_path