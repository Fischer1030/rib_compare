from pathlib import Path


# =====================================================
# 输入数据
# =====================================================


# 完整胸腔模型 F
FULL_CHEST_STL = Path(
    r"D:\ylt\files\台账\人工肋骨选型匹配归档\01-S004\01-S004_full.stl"
)


# 患侧肋骨 A
PATIENT_RIB_A = Path(
    r"D:\ylt\files\台账\人工肋骨选型匹配归档\01-S004\01-S004_R9.stl"
)


# 对侧参考肋骨 B
REFERENCE_RIB_B = Path(
    r"D:\ylt\files\台账\人工肋骨选型匹配归档\01-S004\01-S004_L9.stl"
)


# 肿瘤模型 R
TUMOR_STL = Path(
    r"D:\ylt\files\台账\人工肋骨选型匹配归档\01-S004\01-S004_R9-ZL.stl"
)


# 产品库
IMPLANT_LIBRARY = Path(
    r"G:\database\db_rib\implant_lib\products"
)



# =====================================================
# 输出
# =====================================================


OUTPUT_DIR = Path(
    "rib_compare_result"
)


REGISTRATION_DIR = OUTPUT_DIR / "registration"

RESECTION_DIR = OUTPUT_DIR / "resection"

MATCH_DIR = OUTPUT_DIR / "matching"

REPORT_DIR = OUTPUT_DIR / "report"

RECONSTRUCTION_DIR = OUTPUT_DIR / "reconstruction"



for p in [
    OUTPUT_DIR,
    REGISTRATION_DIR,
    RESECTION_DIR,
    MATCH_DIR,
    REPORT_DIR,
    RECONSTRUCTION_DIR
]:

    p.mkdir(
        parents=True,
        exist_ok=True
    )



# =====================================================
# STL处理
# =====================================================


# 是否保留最大连通区域

KEEP_LARGEST_COMPONENT = True



# 是否修复法向

FIX_NORMAL = True



# =====================================================
# 配准参数
# =====================================================


# ICP最大迭代

ICP_MAX_ITER = 100



# 初始搜索旋转角范围

ROTATION_RANGE = 30



# 配准采样点

REGISTRATION_POINTS = 5000



# =====================================================
# 肿瘤/切除
# =====================================================


# 肿瘤向两侧延伸搭接长度

OVERLAP_LENGTH = 20     # mm



# 切除安全边界

RESECTION_MARGIN = 5    # mm



# =====================================================
# 中心线
# =====================================================


CENTERLINE_POINTS = 300


VOXEL_SIZE = 0.5


SMOOTH_WINDOW = 21

# 人工骨中心线

PRODUCT_CENTERLINE_POINTS = 120

PRODUCT_CENTERLINE_SECTIONS = 180

PRODUCT_CENTERLINE_SMOOTH_WINDOW = 15

PRODUCT_CENTERLINE_SMOOTH_POLYORDER = 3

# =====================================================
# 产品匹配
# =====================================================


# 长度筛选单位

LENGTH_STEP = 30



# 搜索最近几个长度等级

LENGTH_RANGE = 1


# 返回Top数量

TOP_K = 3



# RMSE权重

WEIGHT_RMSE = 1.0

WEIGHT_CURVATURE = 0.3

WEIGHT_LENGTH = 0.2



# =====================================================
# 可视化
# =====================================================


SHOW_COORDINATE = True


SAVE_FIGURE = True



# =====================================================
# 重建
# =====================================================


MERGE_DISTANCE = 0.5
