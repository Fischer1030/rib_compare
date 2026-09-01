import trimesh
import numpy as np



# =====================================================
# STL读取
# =====================================================


def load_stl(path):

    mesh = trimesh.load(
        path,
        force="mesh"
    )

    return mesh




# =====================================================
# 基础清理
# =====================================================


def clean_mesh(
    mesh,
    check_components=True
):

    print()
    print("-" * 60)
    print("网格基础清理")
    print("-" * 60)


    # ========================================================
    # 1. 删除重复顶点
    # ========================================================

    try:

        mesh.merge_vertices()

        print(
            "重复顶点清理完成。"
        )

    except Exception as e:

        print(
            f"重复顶点清理失败：{e}"
        )


    # ========================================================
    # 2. 删除退化面
    # ========================================================

    try:

        mask = trimesh.triangles.area(
            mesh.triangles
        ) > 1e-12


        mesh.update_faces(
            mask
        )

        print(
            "退化面清理完成。"
        )

    except Exception as e:

        print(
            f"退化面清理失败：{e}"
        )


    # ========================================================
    # 3. 删除无引用顶点
    # ========================================================

    try:

        mesh.remove_unreferenced_vertices()

        print(
            "未使用顶点清理完成。"
        )

    except Exception as e:

        print(
            f"未使用顶点清理失败：{e}"
        )


    # ========================================================
    # 4. 修复法向
    # ========================================================

    try:

        trimesh.repair.fix_normals(
            mesh
        )

        print(
            "法向修复完成。"
        )

    except Exception as e:

        print(
            f"法向修复失败：{e}"
        )


    # ========================================================
    # 5. 组件检查
    # ========================================================

    if check_components:

        try:

            components = mesh.split(
                only_watertight=False
            )

            print(
                f"连通组件数量：{len(components)}"
            )


            if len(components) > 1:

                print(
                    "组件面数（前4个）："
                )


                components = sorted(
                    components,
                    key=lambda x: len(x.faces),
                    reverse=True
                )


                for i, c in enumerate(
                    components[:4],
                    start=1
                ):

                    print(
                        f"{i}: "
                        f"{len(c.faces):,} faces"
                    )


        except Exception as e:

            print(
                f"组件检查失败：{e}"
            )


    return mesh

# =====================================================
# 最大连通区域
# =====================================================


def largest_component(mesh):


    parts = mesh.split(
        only_watertight=False
    )


    if len(parts)==1:

        return mesh



    parts.sort(
        key=lambda x:len(x.faces),
        reverse=True
    )


    return parts[0]




# =====================================================
# 尺寸
# =====================================================


def mesh_size(mesh):


    box = mesh.bounding_box.extents


    return {

        "X":box[0],
        "Y":box[1],
        "Z":box[2]

    }





# =====================================================
# 中心点
# =====================================================


def mesh_center(mesh):


    return mesh.centroid




# =====================================================
# 点云采样
# =====================================================


def sample_points(
        mesh,
        n=5000
):


    points,_ = trimesh.sample.sample_surface(
        mesh,
        n
    )


    return points




# =====================================================
# 平移
# =====================================================


def translate_mesh(
        mesh,
        vector
):


    m = mesh.copy()

    m.apply_translation(
        vector
    )

    return m





# =====================================================
# 旋转
# =====================================================


def rotate_mesh(
        mesh,
        matrix
):


    m=mesh.copy()


    m.apply_transform(
        matrix
    )


    return m
