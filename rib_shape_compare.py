from pathlib import Path

import trimesh
import config
import numpy as np
from mesh_utils import (
    load_stl,
    clean_mesh,
    largest_component
)
from product_centerline import (
    extract_product_centerline
)
from registration import (
    register_A_to_B,
    register_implant_to_A2
)

from resection import (
    create_reconstruction_region
)

from centerline import (
    extract_centerline
)

from comparison import (
    centerline_length,
    match_products
)

from visualization import (
    save_registration_report,
    save_centerline_overlap,
    save_mesh_overlap,
    save_three_mesh_overlap
)

from reconstruction import (
    build_reconstruction,
    save_reconstruction
)


# ============================================================
# STL准备
# ============================================================

def prepare_mesh(
    path,
    keep_largest=True
):

    mesh = load_stl(path)

    mesh = clean_mesh(mesh)


    if keep_largest:

        mesh = largest_component(mesh)


    return mesh
# ============================================================
# 产品库读取
# ============================================================

import re


def load_product_library(target_length):

    products = []

    # ========================================================
    # 产品长度预筛选
    #
    # 产品规格：
    # 30 mm 一个主体段
    #
    # 例如：
    # 090 → 90 mm
    # 120 → 120 mm
    # 150 → 150 mm
    # 180 → 180 mm
    # 210 → 210 mm
    # 240 → 240 mm
    #
    # A2长度从 config 中读取
    # ========================================================


    # 产品允许长度范围
    MIN_PRODUCT_LENGTH = 90.0
    MAX_PRODUCT_LENGTH = 240.0

    # 允许的产品规格
    PRODUCT_LENGTHS = list(
        range(
            90,
            241,
            30
        )
    )

    # --------------------------------------------------------
    # 根据 A2 长度确定候选产品长度
    #
    # 这里不是要求产品长度等于 A2，
    # 而是：
    #
    # 产品长度 >= A2长度
    #
    # 后续再由形状匹配决定最佳产品。
    # --------------------------------------------------------

    candidate_lengths = [
        length
        for length in PRODUCT_LENGTHS
        if abs(length - target_length) <= 45
    ]

    # 如果 A2 超过最大产品长度
    if not candidate_lengths:

        raise RuntimeError(
            "\n"
            "没有符合条件的产品长度。\n"
            f"A2中心线长度："
            f"{target_length:.2f} mm\n"
            f"产品最大长度："
            f"{MAX_PRODUCT_LENGTH:.2f} mm"
        )

    print()
    print("-" * 60)
    print("产品长度预筛选")
    print("-" * 60)

    print(
        f"A2中心线长度："
        f"{target_length:.2f} mm"
    )

    print(
        "候选产品长度：",
        candidate_lengths
    )

    # ========================================================
    # 遍历产品
    # ========================================================

    for path in config.IMPLANT_LIBRARY.glob("*.stl"):

        try:

            name = path.stem.upper()

            # ------------------------------------------------
            # 从文件名中提取产品长度
            #
            # 例如：
            #
            # FML01-0901203-BE
            #      ↑
            #     090
            #
            # FML08-2401203-EM
            #      ↑
            #     240
            # ------------------------------------------------

            match = re.search(
                r"-(\d{3})",
                name
            )

            if match is None:

                print(
                    f"跳过产品："
                    f"{path.name} "
                    f"（无法识别长度）"
                )

                continue

            product_length = float(
                match.group(1)
            )

            # ------------------------------------------------
            # 长度预筛选
            # ------------------------------------------------

            if product_length not in candidate_lengths:

                continue

            print()
            print(
                f"进入产品计算："
                f"{path.name}"
            )

            # ------------------------------------------------
            # 只有通过长度筛选之后，
            # 才真正读取 / 处理 STL
            # ------------------------------------------------

            mesh = prepare_mesh(
                path,
                False
            )



            centerline = (
                extract_product_centerline(
                    mesh,
                    n_points=
                    config.PRODUCT_CENTERLINE_POINTS
                )
            )



            centerline_actual_length = centerline_length(
                centerline
            )

            products.append({

                "name": path.stem,

                "path": path,

                "mesh": mesh,

                "centerline": centerline,

                # 文件名中的真实产品长度
                "length": product_length,

                # 实际计算得到的中心线长度
                "centerline_length":
                    centerline_actual_length,

                "side": detect_product_side(
                    name
                )

            })

            print(
                f"产品中心线完成："
                f"{path.name} "
                f"文件名长度={product_length:.2f} mm "
                f"中心线={centerline_actual_length:.2f} mm"
            )

        except Exception as e:

            print(
                f"产品处理失败："
                f"{path.name}"
            )

            print(e)

    print()
    print("-" * 60)
    print(
        f"产品长度预筛选完成："
        f"{len(products)} 个产品进入计算"
    )
    print("-" * 60)

    return products
# ============================================================
# 判断产品左右侧
# ============================================================

def detect_product_side(
    name
):

    name = name.upper()

    if name.startswith("L"):

        return "L"

    if name.startswith("R"):

        return "R"

    if "FML" in name:

        return "L"

    if "FMR" in name:

        return "R"

    return None


# ============================================================
# 根据患侧确定产品侧别
# ============================================================

def detect_patient_side(
    path
):

    name = path.stem.upper()

    if name.startswith("L"):

        return "L"

    if name.startswith("R"):

        return "R"

    if "_L" in name:

        return "L"

    if "_R" in name:

        return "R"

    return None


# ============================================================
# 主程序
# ============================================================

def main():

    print("=" * 70)
    print("Rib Reconstruction Matching System")
    print("=" * 70)

    # ========================================================
    # 1. 读取模型
    # ========================================================

    print()
    print("读取患者完整胸腔 F")

    full_chest = prepare_mesh(
        config.FULL_CHEST_STL,
        True
    )

    print()
    print("读取患侧肋骨 A")

    patient_A = prepare_mesh(
        config.PATIENT_RIB_A,
        True
    )

    print()
    print("读取对侧肋骨 B")

    reference_B = prepare_mesh(
        config.REFERENCE_RIB_B,
        True
    )

    print()
    print("读取肿瘤 R")

    tumor = prepare_mesh(
        config.TUMOR_STL,
        True
    )

    # ========================================================
    # 2. A ↔ B
    # ========================================================

    print()
    print("=" * 70)
    print("A ↔ B 配准")
    print("=" * 70)

    registered_B, T_Bmirror_to_A = register_A_to_B(
        patient_A,
        reference_B
    )

    print()
    print("=" * 70)
    print("最终坐标范围检查")
    print("=" * 70)

    print("patient_A bounds:")
    print(patient_A.bounds)

    print()

    print("tumor bounds:")
    print(tumor.bounds)

    print()

    print("registered_B bounds:")
    print(registered_B.bounds)

    save_registration_report(

        patient_A,

        registered_B,

        config.REGISTRATION_DIR /
        "A_B_registration.png",

        "A",

        "B"

    )

    # ========================================================
    # 3. 肿瘤 / 切除区域
    # ========================================================

    print()
    print("=" * 70)
    print("计算肿瘤对应区域")
    print("=" * 70)

    A1, A2 = create_reconstruction_region(

        patient_A,

        registered_B,

        tumor

    )

    A1.export(
        config.RESECTION_DIR /
        "A1_patient_resection.stl"
    )

    A2.export(
        config.RESECTION_DIR /
        "A2_reference_defect.stl"
    )

    # ========================================================
    # 4. A2中心线
    # ========================================================

    print()
    print("=" * 70)
    print("提取 A2 中心线")
    print("=" * 70)

    A2_centerline = extract_centerline(

        A2,

        voxel_size=config.VOXEL_SIZE,

        n_points=config.CENTERLINE_POINTS

    )

    A2_length = centerline_length(
        A2_centerline
    )

    print(
        f"A2中心线长度："
        f"{A2_length:.2f} mm"
    )

    if A2_length > 270.0:
        raise RuntimeError(
            f"\n"
            f"A2中心线长度异常：{A2_length:.2f} mm\n"
            f"A2最大几何尺寸："
            f"{np.max(A2.bounding_box.extents):.2f} mm\n"
            f"超过固定上限 270 mm，"
            f"停止后续产品匹配。"
        )

    from resection import save_a2_model
    save_a2_model(
        A2,
        config.OUTPUT_DIR /
        "A2_check.stl"
    )

    # ========================================================
    # 5. 确定左右侧
    # ========================================================

    patient_side = detect_patient_side(
        config.PATIENT_RIB_A
    )

    if patient_side is None:

        raise RuntimeError(
            "无法从患者肋骨文件名判断L/R侧。"
        )

    print(
        f"患侧：{patient_side}"
    )

    # ========================================================
    # 6. 产品库
    # ========================================================

    print()
    print("=" * 70)
    print("读取产品库")
    print("=" * 70)

    products = load_product_library(A2_length)

    print(
        f"产品总数量："
        f"{len(products)}"
    )

    # ========================================================
    # 7. 产品侧别过滤
    # ========================================================

    products = [

        p

        for p in products

        if p["side"] == patient_side

    ]

    print(
        f"{patient_side}系列产品："
        f"{len(products)}"
    )

    # ========================================================
    # 8. 长度 + 形态匹配
    # ========================================================

    results = match_products(

        A2_centerline,

        products,

        topk=config.TOP_K

    )

    # ========================================================
    # 9. 输出Top3
    # ========================================================

    print()
    print("=" * 70)
    print("Top 3 产品")
    print("=" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        print()
        print(
            f"#{i} "
            f"{result['name']}"
        )

        print(
            f"长度："
            f"{result['length']:.2f} mm"
        )

        print(
            f"RMSE："
            f"{result['RMSE']:.4f}"
        )

        print(
            f"曲率差："
            f"{result['curvature']:.6f}"
        )

        print(
            f"综合评分："
            f"{result['score']:.6f}"
        )

    if not results:

        raise RuntimeError(
            "没有找到符合长度和侧别要求的产品。"
        )

    # ========================================================
    # 10. 操作者选择
    # ========================================================

    print()
    print("=" * 70)
    print("请选择最终产品")
    print("=" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{i}. "
            f"{result['name']}"
        )

    while True:

        choice = input(
            "请输入产品编号："
        )

        try:

            index = int(choice)

            if 1 <= index <= len(results):

                break

        except ValueError:

            pass

        print(
            "输入无效，请重新输入。"
        )

    selected = results[
        index - 1
    ]

    print()
    print(
        f"最终选择："
        f"{selected['name']}"
    )

    # ========================================================
    # 11. 产品 ↔ A2 三轴配准
    # ========================================================

    implant_mesh = selected[
        "mesh"
    ]

    registered_implant, T_AP = (
        register_implant_to_A2(

            implant_mesh,

            A2

        )
    )

    # ========================================================
    # 12. 产品配准结果
    # ========================================================

    save_mesh_overlap(

        A2,

        registered_implant,

        config.MATCH_DIR /
        f"{selected['name']}_A2_registration.png",

        "A2",

        selected["name"],

        "A2 ↔ Implant"

    )

    implant_centerline = extract_product_centerline(

        registered_implant,

        n_points=config.PRODUCT_CENTERLINE_POINTS

    )

    save_centerline_overlap(

        A2_centerline,

        implant_centerline,

        config.MATCH_DIR /
        f"{selected['name']}_centerline.png",

        "A2",

        selected["name"]

    )

    # ========================================================
    # 13. 最终重建
    # ========================================================

    print()
    print("=" * 70)
    print("生成最终重建模型")
    print("=" * 70)

    reconstruction = build_reconstruction(

        full_chest,

        A1,

        registered_implant

    )

    output_path = (

        config.RECONSTRUCTION_DIR /

        "final_reconstruction.stl"

    )

    save_reconstruction(

        reconstruction,

        output_path

    )

    # ========================================================
    # 14. 三模型显示
    # ========================================================

    save_three_mesh_overlap(

        full_chest,

        A1,

        registered_implant,

        config.RECONSTRUCTION_DIR /
        "final_reconstruction_overlap.png",

        labels=(
            "F",

            "A1",

            selected["name"]

        )

    )

    print()
    print("=" * 70)
    print("全部流程完成")
    print("=" * 70)





# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()