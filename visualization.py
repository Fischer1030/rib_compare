from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import trimesh


# ============================================================
# 内部：创建3D坐标轴
# ============================================================

def _create_figure(title=None):

    fig = plt.figure(
        figsize=(10, 8)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    if title:
        ax.set_title(title)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    return fig, ax


# ============================================================
# 绘制中心线
# ============================================================

def _plot_centerline(
    ax,
    centerline,
    label,
    linewidth=2.0
):

    centerline = np.asarray(
        centerline
    )

    ax.plot(
        centerline[:, 0],
        centerline[:, 1],
        centerline[:, 2],
        linewidth=linewidth,
        label=label
    )


# ============================================================
# 设置等比例坐标
# ============================================================

def _set_equal_axes(ax, points):

    points = np.asarray(points)

    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    centers = (mins + maxs) / 2.0

    radius = np.max(
        maxs - mins
    ) / 2.0

    if radius <= 0:
        radius = 1.0

    ax.set_xlim(
        centers[0] - radius,
        centers[0] + radius
    )

    ax.set_ylim(
        centers[1] - radius,
        centers[1] + radius
    )

    ax.set_zlim(
        centers[2] - radius,
        centers[2] + radius
    )


# ============================================================
# STL表面绘制
# ============================================================

def _plot_mesh(
    ax,
    mesh,
    label,
    alpha=0.35,
    max_faces=15000
):

    mesh = mesh.copy()

    # 面数过多时仅用于显示
    if len(mesh.faces) > max_faces:

        try:

            mesh = mesh.simplify_quadric_decimation(
                max_faces
            )

        except Exception:

            pass

    vertices = np.asarray(
        mesh.vertices
    )

    faces = np.asarray(
        mesh.faces
    )

    if len(vertices) == 0 or len(faces) == 0:
        return

    collection = ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        alpha=alpha
    )

    collection.set_label(label)


# ============================================================
# 中心线重叠显示
# ============================================================

def save_centerline_overlap(
    patient_centerline,
    implant_centerline,
    output_path,
    patient_label="A2",
    implant_label="Product"
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = _create_figure(
        "Centerline Registration"
    )

    _plot_centerline(
        ax,
        patient_centerline,
        patient_label,
        linewidth=3
    )

    _plot_centerline(
        ax,
        implant_centerline,
        implant_label,
        linewidth=3
    )

    points = np.vstack([
        patient_centerline,
        implant_centerline
    ])

    _set_equal_axes(
        ax,
        points
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)


# ============================================================
# STL重叠显示
# ============================================================

def save_mesh_overlap(
    mesh_a,
    mesh_b,
    output_path,
    label_a="A2",
    label_b="Product",
    title="STL Registration"
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = _create_figure(
        title
    )

    _plot_mesh(
        ax,
        mesh_a,
        label_a,
        alpha=0.45
    )

    _plot_mesh(
        ax,
        mesh_b,
        label_b,
        alpha=0.45
    )

    points = np.vstack([
        mesh_a.vertices,
        mesh_b.vertices
    ])

    _set_equal_axes(
        ax,
        points
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)


# ============================================================
# A/B配准报告
# ============================================================

def save_registration_report(
    source_mesh,
    target_mesh,
    output_path,
    source_label="A",
    target_label="B"
):

    save_mesh_overlap(
        source_mesh,
        target_mesh,
        output_path,
        source_label,
        target_label,
        title=f"{source_label} ↔ {target_label} Registration"
    )


# ============================================================
# 三模型重叠
# ============================================================

def save_three_mesh_overlap(
    mesh_a,
    mesh_b,
    mesh_p,
    output_path,
    labels=("F", "A1", "Product")
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fig, ax = _create_figure(
        "Final Reconstruction Registration"
    )

    _plot_mesh(
        ax,
        mesh_a,
        labels[0],
        alpha=0.25
    )

    _plot_mesh(
        ax,
        mesh_b,
        labels[1],
        alpha=0.45
    )

    _plot_mesh(
        ax,
        mesh_p,
        labels[2],
        alpha=0.45
    )

    points = np.vstack([
        mesh_a.vertices,
        mesh_b.vertices,
        mesh_p.vertices
    ])

    _set_equal_axes(
        ax,
        points
    )

    ax.legend()

    plt.tight_layout()

    fig.savefig(
        output_path,
        dpi=300
    )

    plt.close(fig)