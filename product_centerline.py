import numpy as np
import trimesh

from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull


# ============================================================
# 人工骨中心线提取
# ============================================================


def extract_product_centerline(
        mesh,
        n_points=120,
        n_sections=180,
        smooth_window=11,
        smooth_polyorder=3
):
    """
    人工肋骨产品中心线

    方法：

    STL
     ↓
    PCA主方向
     ↓
    等距截面
     ↓
    截面几何中心
     ↓
    Savgol平滑
     ↓
    弧长重采样

    返回:

        N×3 centerline
    """

    if not isinstance(
        mesh,
        trimesh.Trimesh
    ):
        raise TypeError(
            "输入必须为 trimesh.Trimesh"
        )


    vertices = np.asarray(
        mesh.vertices,
        dtype=np.float64
    )


    # ==============================
    # PCA
    # ==============================

    center = vertices.mean(
        axis=0
    )


    centered = (
        vertices-center
    )


    cov = np.cov(
        centered,
        rowvar=False
    )


    eigval,eigvec=np.linalg.eigh(
        cov
    )


    axis = eigvec[
        :,
        np.argmax(eigval)
    ]


    axis /= np.linalg.norm(axis)


    # ==============================
    # 建立截面坐标
    # ==============================


    ref=np.array(
        [0,0,1],
        dtype=float
    )


    if abs(
        np.dot(
            ref,
            axis
        )
    )>0.9:

        ref=np.array(
            [0,1,0],
            dtype=float
        )


    u=np.cross(
        axis,
        ref
    )

    u/=np.linalg.norm(u)


    v=np.cross(
        axis,
        u
    )

    v/=np.linalg.norm(v)



    # ==============================
    # 主轴投影
    # ==============================


    projection=(
        centered @ axis
    )


    s_min=np.min(
        projection
    )

    s_max=np.max(
        projection
    )


    sections=np.linspace(
        s_min,
        s_max,
        n_sections
    )


    raw=[]

    valid=[]


    width=(
        s_max-s_min
    )/n_sections*0.8



    # ==============================
    # 截面中心
    # ==============================


    for s in sections:


        mask=np.abs(
            projection-s
        )<=width


        pts=vertices[
            mask
        ]


        if len(pts)<3:
            continue


        relative=(
            pts-
            (
                center+
                axis*s
            )
        )


        uv=np.column_stack(
            [
                relative@u,
                relative@v
            ]
        )


        try:

            hull=ConvexHull(
                uv
            )

            uv=uv[
                hull.vertices
            ]

        except:

            pass


        c2=uv.mean(
            axis=0
        )


        p=(
            center+
            axis*s+
            u*c2[0]+
            v*c2[1]
        )


        raw.append(p)

        valid.append(s)



    raw=np.asarray(raw)


    if len(raw)<10:

        raise RuntimeError(
            "有效截面不足"
        )


    # ==============================
    # 排序
    # ==============================

    order=np.argsort(valid)

    raw=raw[order]


    # ==============================
    # 平滑
    # ==============================

    smooth=smooth_centerline(
        raw,
        smooth_window,
        smooth_polyorder
    )


    # ==============================
    # 弧长重采样
    # ==============================


    length=np.linalg.norm(
        np.diff(
            smooth,
            axis=0
        ),
        axis=1
    )


    cumulative=np.concatenate(
        [
            [0],
            np.cumsum(length)
        ]
    )


    total=cumulative[-1]


    target=np.linspace(
        0,
        total,
        n_points
    )


    centerline=np.column_stack(
        [
            np.interp(
                target,
                cumulative,
                smooth[:,i]
            )
            for i in range(3)
        ]
    )


    centerline=smooth_centerline(
        centerline,
        15,
        3
    )


    return centerline



# ============================================================
# 平滑
# ============================================================


def smooth_centerline(
        points,
        window=15,
        polyorder=3
):


    points=np.asarray(
        points,
        dtype=float
    )


    if len(points)<window:
        return points


    start=points[0].copy()
    end=points[-1].copy()


    result=np.zeros_like(
        points
    )


    for i in range(3):

        result[:,i]=savgol_filter(
            points[:,i],
            window,
            polyorder
        )


    result[0]=start
    result[-1]=end


    return result