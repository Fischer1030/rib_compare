import numpy as np
import trimesh

from scipy.spatial.transform import Rotation

import open3d as o3d



# =====================================================
# mesh 转 open3d
# =====================================================

# =====================================================
# 应用刚体变换
# =====================================================

def apply_transform(
        mesh,
        transform
):

    result = mesh.copy()

    result.apply_transform(
        transform
    )

    return result

def trimesh_to_open3d(mesh):


    vertices = np.asarray(
        mesh.vertices
    )

    triangles = np.asarray(
        mesh.faces
    )


    o3mesh = o3d.geometry.TriangleMesh()


    o3mesh.vertices = (
        o3d.utility.Vector3dVector(
            vertices
        )
    )


    o3mesh.triangles = (
        o3d.utility.Vector3iVector(
            triangles
        )
    )


    return o3mesh




# =====================================================
# 镜像
# =====================================================


def mirror_mesh(
        mesh,
        axis="X"
):


    m=mesh.copy()


    vertices=m.vertices.copy()


    if axis=="X":

        vertices[:,0]*=-1


    elif axis=="Y":

        vertices[:,1]*=-1


    elif axis=="Z":

        vertices[:,2]*=-1


    m.vertices=vertices


    return m





# =====================================================
# PCA 初始对齐
# =====================================================


def pca_align(
        source,
        target
):


    src=np.asarray(
        source.vertices
    )


    tgt=np.asarray(
        target.vertices
    )



    src_center=src.mean(axis=0)

    tgt_center=tgt.mean(axis=0)



    src0=src-src_center

    tgt0=tgt-tgt_center



    src_cov=np.cov(
        src0.T
    )


    tgt_cov=np.cov(
        tgt0.T
    )


    src_vec=np.linalg.eigh(
        src_cov
    )[1][:,::-1]


    tgt_vec=np.linalg.eigh(
        tgt_cov
    )[1][:,::-1]



    R=tgt_vec@src_vec.T



    if np.linalg.det(R)<0:

        R[:,2]*=-1



    T=np.eye(4)


    T[:3,:3]=R


    T[:3,3]=(
        tgt_center -
        R@src_center
    )



    result=source.copy()


    result.apply_transform(
        T
    )


    return result,T





# =====================================================
# ICP 精配准
# =====================================================


def icp_registration(
        source,
        target,
        max_iter=100
):


    src=o3d.geometry.PointCloud()

    tgt=o3d.geometry.PointCloud()



    src.points=o3d.utility.Vector3dVector(
        source.vertices
    )


    tgt.points=o3d.utility.Vector3dVector(
        target.vertices
    )



    result=o3d.pipelines.registration.registration_icp(

        src,

        tgt,

        5.0,

        np.eye(4),

        o3d.pipelines.registration.TransformationEstimationPointToPoint(),

        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iter
        )

    )



    aligned=source.copy()


    aligned.apply_transform(
        result.transformation
    )



    return aligned,result.transformation





# =====================================================
# A ↔ B 配准
# =====================================================


def register_A_to_B(
        patient_A,
        rib_B
):


    # B镜像到A侧

    B_mirror=mirror_mesh(
        rib_B,
        "X"
    )


    # PCA

    init,T1=pca_align(
        B_mirror,
        patient_A
    )


    # ICP

    aligned,T2=icp_registration(
        init,
        patient_A
    )



    return aligned,T1@T2





# =====================================================
# 产品 ↔ A2
# =====================================================


def register_implant_to_A2(
        implant,
        A2
):


    init,T1=pca_align(
        implant,
        A2
    )


    aligned,T2=icp_registration(
        init,
        A2
    )


    return aligned,T1@T2
