import numpy as np
import trimesh
import networkx as nx

from scipy.ndimage import binary_fill_holes

from scipy.signal import savgol_filter

from skimage.morphology import skeletonize

from scipy.spatial import distance_matrix



# =====================================================
# STL体素化
# =====================================================


def voxelize_mesh(
        mesh,
        voxel_size=0.5
):


    vox = mesh.voxelized(
        voxel_size
    )


    volume = vox.matrix.copy()


    transform = vox.transform


    return volume,transform




# =====================================================
# skeleton
# =====================================================


def skeleton_3d(
        volume
):


    volume=binary_fill_holes(
        volume
    )

    from scipy import ndimage

    print()
    print("-" * 60)
    print("Skeletonize 前体素连通性检查")
    print("-" * 60)

    voxel_labels, voxel_components = ndimage.label(
        volume
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


    skeleton=skeletonize(
        volume
    )

    from scipy import ndimage

    structure = ndimage.generate_binary_structure(
        3,
        3
    )

    labels, num_components = ndimage.label(
        skeleton,
        structure=structure
    )

    print(
        f"Skeleton 26邻域连通组件："
        f"{num_components}"
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
        f"Skeleton连通组件："
        f"{num_components}"
    )


    return skeleton




# =====================================================
# skeleton坐标
# =====================================================


def skeleton_points(
        skeleton,
        transform
):


    idx=np.argwhere(
        skeleton
    )


    points=[]


    for p in idx:


        xyz=trimesh.transform_points(

            np.array(
                [p[::-1]]
            ),

            transform

        )[0]


        points.append(
            xyz
        )


    return np.array(points)





# =====================================================
# 最长路径
# 简化版本
# =====================================================


def longest_path(
    skeleton,
    voxel_size,
    transform
):



    # ===============================
    # skeleton voxel 坐标
    # ===============================

    coords = np.argwhere(
        skeleton
    )


    n = len(coords)


    print(
        f"Skeleton节点数量：{n}"
    )


    # ===============================
    # 建立26邻域图
    # ===============================

    G = nx.Graph()


    for i in range(n):

        G.add_node(i)



    lookup = {

        tuple(c):i

        for i,c in enumerate(coords)

    }



    neighbor_offsets=[]


    for x in [-1,0,1]:

        for y in [-1,0,1]:

            for z in [-1,0,1]:

                if (
                    x==0 and
                    y==0 and
                    z==0
                ):
                    continue


                neighbor_offsets.append(
                    np.array(
                        [x,y,z]
                    )
                )



    for i,c in enumerate(coords):

        for offset in neighbor_offsets:


            nb=tuple(
                c+offset
            )


            if nb in lookup:


                j=lookup[nb]


                if i<j:

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
                        weight=distance
                    )



    print(
        "Skeleton图节点:",
        G.number_of_nodes()
    )


    print(
        "Skeleton图边:",
        G.number_of_edges()
    )


    # ===============================
    # 找最长路径
    # ===============================


    # 第一次：
    # 任意点

    start=0


    dist,_ = nx.single_source_dijkstra(
        G,
        start
    )


    end=max(
        dist,
        key=dist.get
    )


    # 第二次

    dist,path = nx.single_source_dijkstra(
        G,
        end
    )


    end2=max(
        dist,
        key=dist.get
    )


    node_path = nx.shortest_path(
        G,
        end,
        end2,
        weight="weight"
    )


    print(
        "最长路径节点:",
        len(node_path)
    )


    print(
        "骨架长度:",
        dist[end2]
    )



    # ===============================
    # voxel -> 世界坐标
    # ===============================


    centerline=[]


    for i in node_path:


        p=coords[i]


        xyz=trimesh.transform_points(

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


    return np.array(centerline)




# =====================================================
# 平滑
# =====================================================


def smooth_centerline(
        line,
        window=21
):


    if len(line)<window:

        return line



    result=np.zeros_like(
        line
    )


    for i in range(3):


        result[:,i]=savgol_filter(

            line[:,i],

            window,

            3

        )


    return result






# =====================================================
# 重采样
# =====================================================


def resample_centerline(
        line,
        n_points=300
):


    length=np.zeros(
        len(line)
    )


    for i in range(
        1,
        len(line)
    ):


        length[i]=(

            length[i-1]

            +

            np.linalg.norm(

                line[i]-line[i-1]

            )

        )



    new_length=np.linspace(

        0,

        length[-1],

        n_points

    )


    result=[]



    for l in new_length:


        idx=np.searchsorted(
            length,
            l
        )


        if idx==0:

            result.append(
                line[0]
            )


        else:

            ratio=(

                l-length[idx-1]

            )/(

                length[idx]-length[idx-1]

            )


            p=(

                line[idx-1]

                +

                ratio*

                (

                line[idx]-line[idx-1]

                )

            )


            result.append(p)



    return np.array(result)






# =====================================================
# 外部调用
# =====================================================


def extract_centerline(
        mesh,
        voxel_size=0.5,
        n_points=300
):


    volume,transform=voxelize_mesh(

        mesh,

        voxel_size

    )


    skeleton=skeleton_3d(
        volume
    )


    pts=skeleton_points(

        skeleton,

        transform

    )

    line = longest_path(
        skeleton,
        voxel_size,
        transform
    )


    line=smooth_centerline(
        line
    )


    line=resample_centerline(

        line,

        n_points

    )


    return line
