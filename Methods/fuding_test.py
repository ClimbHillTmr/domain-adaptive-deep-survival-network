import sys

sys.path.append("..")
import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split
# from Methods.Models.LightGBM_model import LightGBM_model
from sklearn.utils.class_weight import compute_class_weight


def fuding_test():

    test_dataset = pd.read_csv(
            "/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/福鼎_final_data.csv"
        )

    label_dict = {
        "传染病": {
            "丙肝,": 0,
            "乙肝,": 1,
            "乙肝,丙肝,": 2,
            "乙肝,梅毒,": 3,
            "梅毒,": 4,
            "阴性": 5,
            np.nan: 6,
        },
        # "性别": {"女": 0, "男": 1, np.nan: 2},
        # "患者状态": {
        #     "其他": 0,
        #     "出院": 1,
        #     "在透": 2,
        #     "好转": 3,
        #     "放弃治疗": 4,
        #     "死亡": 5,
        #     "肾移植": 6,
        #     "腹膜透析": 7,
        #     "转院": 8,
        #     np.nan: 9,
        # },
        # "透析器": {
        #     "15AC": 0,
        #     "15LC": 1,
        #     "15UC": 2,
        #     "18AC": 3,
        #     "18C": 4,
        #     "18LC": 5,
        #     "18UC": 6,
        #     "AV600S": 7,
        #     "F16": 8,
        #     "FX80": 9,
        #     "中截流量中空纤维透析器": 10,
        #     "百特R300": 11,
        #     "百特R400": 12,
        #     "血浆滤过器": 13,
        #     np.nan: 14,
        # },
        # 抗凝剂类型：抗凝剂为枸盐酸钠和阿加曲班的字段均改为其他，编号0，即深医中 0其他，1低分子肝素，2无肝素，3普通肝素，4 nan
        "抗凝剂类型": {"4%枸橼酸钠": 0, "低分子肝素": 1, "无肝素": 2, "普通肝素": 3, "阿加曲班": 0, np.nan: 4},
        "透析方式": {"HD": 0, "HDF": 1, "HF": 2, "HP": 3, "UF": 4, np.nan: 5},
        "瘘管类型": {
            "中心静脉临时导管": 0,
            "中心静脉长期导管": 1,
            "人工血管内瘘": 2,
            "直穿": 3,
            "自体动静脉内瘘": 4,
            np.nan: 5,
        },
        "瘘管位置": {"右上肢": 0, "右下肢": 1, "右颈部": 2, "左上肢": 3, "左下肢": 4, "左颈部": 5, np.nan: 6},
    }
    # 使用label_dict字典替换DataFrame中的列值
    for column, mapping in label_dict.items():
        if column in test_dataset.columns:
            test_dataset[column] = test_dataset[column].map(mapping)

    return test_dataset
