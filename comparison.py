import numpy as np





# =====================================================
# 长度
# =====================================================
import re
from pathlib import Path


def parse_product_name(
    filepath
):
    """
    解析产品型号

    示例：
    FMR10-2401203-BJ.stl


    返回：

    {
        side:"R",
        rib:10,
        length:240
    }

    """

    name = Path(filepath).stem.upper()


    result = {

        "side":None,

        "rib":None,

        "length":None,

        "path":filepath
    }


    # ==============================
    # 左右侧
    # ==============================

    if name.startswith("FMR"):

        result["side"]="R"


    elif name.startswith("FML"):

        result["side"]="L"


    else:

        return None



    # ==============================
    # 肋骨编号
    # FMR10
    # ==============================

    rib_match = re.search(
        r"FM[LR](\d+)",
        name
    )


    if rib_match:

        result["rib"] = int(
            rib_match.group(1)
        )



    # ==============================
    # 长度
    # 2401203
    # 第一个三位数字
    # ==============================

    length_match = re.search(
        r"-([0-9]{3})",
        name
    )


    if length_match:

        result["length"] = int(
            length_match.group(1)
        )


    return result

def filter_product_candidates(
    product_files,
    target_side,
    target_length
):

    """
    产品初筛

    """

    base = (
        round(
            target_length/30
        )
        *30
    )


    allowed_lengths=[

        base-30,

        base,

        base+30

    ]


    candidates=[]


    for file in product_files:


        info=parse_product_name(
            file
        )


        if info is None:

            continue

        # --------------------------
        # 侧别
        # --------------------------

        if info["side"] != target_side:

            continue

        # --------------------------
        # 长度
        # --------------------------

        if info["length"] not in allowed_lengths:

            continue

        candidates.append(
            info
        )


    return candidates

def centerline_length(
        line
):

    return np.sum(

        np.linalg.norm(

            np.diff(
                line,
                axis=0
            ),

            axis=1

        )

    )


# =====================================================
# 归一化
# =====================================================

def normalize_centerline(
        line
):

    length=centerline_length(
        line
    )

    x=np.linspace(

        0,

        length,

        len(line)

    )


    return line



# =====================================================
# RMSE
# =====================================================

def centerline_rmse(
        a,
        b
):


    if len(a)!=len(b):

        n=min(
            len(a),
            len(b)
        )

        a=a[:n]

        b=b[:n]



    return np.sqrt(

        np.mean(

            np.sum(

                (a-b)**2,

                axis=1

            )

        )

    )
def resample_centerline(
        points,
        n_points=120
):
    """
    将中心线按弧长重采样
    """

    points = np.asarray(
        points,
        dtype=float
    )


    segment = np.linalg.norm(
        np.diff(
            points,
            axis=0
        ),
        axis=1
    )


    length = np.concatenate(
        [
            [0],
            np.cumsum(segment)
        ]
    )


    target = np.linspace(
        0,
        length[-1],
        n_points
    )


    result = np.column_stack(
        [
            np.interp(
                target,
                length,
                points[:,i]
            )
            for i in range(3)
        ]
    )


    return result

# =====================================================
# 曲率
# =====================================================


def curvature(
        line
):


    d1=np.gradient(
        line,
        axis=0
    )


    d2=np.gradient(
        d1,
        axis=0
    )


    k=np.linalg.norm(
        d2,
        axis=1
    )


    return k






# =====================================================
# 长度过滤
# =====================================================


def filter_by_length(
        products,
        target_length,
        step=30
):


    center=round(

        target_length/step

    )*step



    valid=[]


    for p in products:


        L=p["length"]



        if abs(
            L-center
        )<=step:


            valid.append(p)



    return valid





# =====================================================
# 产品匹配
# =====================================================

def match_products(
        A2_centerline,
        products,
        topk=3
):

    A2_centerline = resample_centerline(
        A2_centerline,
        120
    )



    target_len=centerline_length(
        A2_centerline
    )


    candidates=filter_by_length(

        products,

        target_len

    )



    results=[]

    for p in candidates:
        product_centerline = resample_centerline(
            p["centerline"],
            120
        )

        rmse = centerline_rmse(
            A2_centerline,
            product_centerline
        )

        curv = np.mean(
            abs(
                curvature(
                    A2_centerline
                )
                -
                curvature(
                    product_centerline
                )
            )
        )

        score=(

            0.01 * rmse

            +

            100 * curv

        )

        results.append({

            "name": p["name"],

            "length": p["length"],

            "RMSE": rmse,

            "curvature": curv,

            "score": score,

            # 保留产品模型
            "mesh": p["mesh"],

            # 保留产品中心线
            "centerline": p["centerline"],

            # 保留产品路径
            "path": p["path"],

            # 保留实际中心线长度
            "centerline_length":
                p.get(
                    "centerline_length",
                    centerline_length(
                        p["centerline"]
                    )
                ),

            # 保留产品侧别
            "side": p.get(
                "side"
            )

        })



    results.sort(

        key=lambda x:x["score"]

    )



    return results[:topk]
