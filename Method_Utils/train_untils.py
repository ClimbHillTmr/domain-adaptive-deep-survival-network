import pandas as pd
import numpy as np


def remove_rows_with_few_duplicates(df):
    # 统计每个值出现的次数
    count_series = df["患者id"].value_counts()

    # 选择出现次数大于等于2的值
    valid_values = count_series[count_series >= 2].index
    print(count_series[count_series < 2].index)

    # 保留原始数据框中值在valid_values中的行
    result_df = df[df["患者id"].isin(valid_values)]

    return result_df


def move_columns_down(df, columns_to_move, column_name="患者id", num_rows=1):
    # 根据患者ID排序数据框
    df_sorted = df.sort_values(by=[column_name])

    # 移动指定数量的行
    df_sorted["移动后索引"] = df_sorted.groupby(column_name).cumcount() + num_rows

    # 将移动后的索引对齐到原始数据框
    df_result = pd.merge(
        df,
        df_sorted[[column_name, "移动后索引"]],
        how="left",
        left_on=column_name,
        right_on=column_name,
    )

    # 根据移动后的索引重新排序
    df_result = df_result.sort_values(by=["移动后索引"]).drop(columns=["移动后索引"])

    # 创建新的数据框，避免修改原始数据框
    new_df = pd.DataFrame()

    # 复制其他列
    for col in df.columns:
        if col not in columns_to_move:
            new_df[col] = df_result[col]

    # 移动指定列的值
    for col in columns_to_move:
        new_df[col] = df_result.groupby(column_name)[col].shift(num_rows)

    return new_df


def create_moved_dataframe(df, columns_to_move, column_name="患者id", num_rows=1):
    # 创建新的数据框，复制原始数据框的内容
    new_df = df.copy()

    # 根据患者ID排序新的数据框
    # new_df = new_df.sort_values(by=[column_name])

    # 使用 groupby 和 shift 在每个组内进行平移
    for col in columns_to_move:
        new_df[col] = new_df.groupby(column_name)[col].shift(num_rows)

    return new_df


def process_dataset(df, columns_to_shift):
    # 读取原始数据集

    # 按照患者id和透析日期排序
    df.sort_values(by=["患者id", "透析日期"], inplace=True)

    # 将每个患者id块的第一行的部分列下移一行
    for column in columns_to_shift:
        df[f"新_{column}"] = df.groupby("患者id")[column].shift(-1)

    # 删除第一行
    df = df.dropna(subset=[f"新_{columns_to_shift[0]}"])

    # 删除不需要的列
    # df = df.drop(columns=columns_to_shift)

    # 重新组合成新的数据集
    new_df = df.groupby("患者id").apply(lambda x: x.reset_index(drop=True))

    return new_df


def process_and_print_stats(dataset):
    # 初始数据集行数和[id]列唯一值数量
    initial_rows = len(dataset)
    initial_unique_id_count = dataset["患者id"].nunique()

    # 输出初始数据集信息
    print(f"初始数据集行数: {initial_rows}")
    print(f"[id]列的唯一值数量: {initial_unique_id_count}")


def Align_standard(dataset):
    dataset = dataset[(dataset["干体重"] > 0) & (dataset["实际透析时长"] > 0)]
    process_and_print_stats(dataset)
    dataset = dataset[(dataset["透析年龄"] >= 16) & (dataset["透析年龄"] <= 100)]
    process_and_print_stats(dataset)
    dataset = dataset[(dataset["实际透析时长"] <= 4) & (dataset["实际透析时长"] >= 3)]
    process_and_print_stats(dataset)
    dataset = dataset[dataset["透中高血压_计算"].notna()]
    process_and_print_stats(dataset)

    # 计算每列的缺失值比例
    missing_percentage = dataset.isnull().mean()
    # print(missing_percentage)

    # 选择缺失值比例小于等于35%的列
    selected_columns = missing_percentage[missing_percentage <= 0.35].index
    # print(missing_percentage[missing_percentage > 0.2])
    
    # # 确保重要的目标列不被删除
    # important_columns = ['透中低血压_计算', '透中高血压_计算', '降幅时间点比值区间', '降幅时间点差值区间']
    # for col in important_columns:
    #     if col in dataset.columns and col not in selected_columns:
    #         selected_columns = selected_columns.union([col])

    # 仅保留选择的列
    df = dataset[selected_columns]

    process_and_print_stats(df)
    return df


def quantile_99(dataset, up=0.99, down=0.01):
    for col in dataset.columns:
        try:
            if str(dataset[col].dtype) == "float64":
                factor = dataset[col]
                mean = factor.mean()
                std = factor.std()

                # 计算上下限的数据
                up_scale = mean + 3 * std
                down_scale = mean - 3 * std

                # up_scale = np.percentile(factor, up)
                # down_scale = np.percentile(factor, down)

                factor = np.where(factor > up_scale, up_scale, factor)
                factor = np.where(factor < down_scale, down_scale, factor)
                dataset[col] = factor
        except Exception as e:
            print(f"处理列 {col} 时出错: {e}")
            continue
    return dataset
    # X = X.astype('float64')


def Iterate_columns(X):
    # Iterate through columns
    for col in X.columns:
        # Check if the column has missing values
        if X[col].isnull().any():
            # Check if the column data type is integer
            if X[col].dtype == "int64":
                X[col].fillna(X[col].mode().iloc[0], inplace=True)
            # Check if the column data type is float
            elif X[col].dtype == "float64":
                X[col].fillna(round(X[col].mean(), 3), inplace=True)
                # Check if the column has missing values

    return X


import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def calculate_class_weights(y):
    classes = np.unique(y)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    return class_weight_dict


features_based = [
    "患者id",
    "透析日期",
    "透前体重-干体重",
    "历史平均透前体重-干体重",
    "性别",
    "透析龄",
    "传染病",
    "透前收缩压",
    "透前舒张压",
    "透后收缩压",
    "透后舒张压",
    "透析龄_天数",
    "透析年龄",
    "透析方式",
    "透前体重",
    "透前呼吸频率",
    "干体重",
    "透析液钙浓度",
    "透析液电导率",
    "抗凝剂类型",
    "透析后体重",
    "实际透析时长",
    "瘘管类型",
    "瘘管位置",
    "瘘管使用时间",
    "首次透析年龄",
    "透析中收缩压_mean",
    "透析中收缩压_std",
    "透析中舒张压_mean",
    "透析中舒张压_std",
    "透析中脉搏_mean",
    "透析中脉搏_std",
    "超滤率_mean",
    "超滤率_std",
    "超滤量MAX",
    "静脉压_mean",
    "静脉压_std",
    "动脉压_mean",
    "动脉压_std",
    "血流速_mean",
    "血流速_std",
    "透析液温度_mean",
    "透析液温度_std",
    "跨膜压_mean",
    "跨膜压_std",
    "历史平均透前体重",
    "历史平均透前呼吸频率",
    "历史平均透前体温",
    "历史平均干体重",
    "历史平均透析液钙浓度",
    "历史平均透析液电导率",
    "历史平均实际透析时长",
    "历史平均透前收缩压",
    "历史平均透前舒张压",
    "历史平均涨幅时间点比值区间",
    "历史平均涨幅时间点差值区间",
    "历史平均涨幅时间点差值",
    "历史平均透中高血压_计算",
    "历史平均透中低血压_计算",
    "历史平均降幅时间点比值区间",
    "历史平均降幅时间点差值区间",
    "历史平均透析中收缩压_mean",
    "历史平均透析中舒张压_mean",
    "历史平均透析中脉搏_mean",
    "历史平均超滤率_mean",
    "历史平均静脉压_mean",
    "历史平均动脉压_mean",
    "历史平均血流速_mean",
    "历史平均透析液温度_mean",
    "历史平均跨膜压_mean",
    "透中高血压_计算",
    "透中低血压_计算",
    "历史平均透中低血压_计算",
    "涨幅时间点比值",
    "涨幅时间点比值区间",
    "涨幅时间点差值",
    "涨幅时间点差值区间",
    "降幅时间点比值区间",
    "降幅时间点差值区间",
]

features = [
    "透前体重-干体重",
    "历史平均透前体重-干体重",
    "性别",
    "透析龄",
    "传染病",
    "透前收缩压",
    "透前舒张压",
    "透后收缩压",
    "透后舒张压",
    "透析龄_天数",
    "透析年龄",
    "透析方式",
    "透前体重",
    "透前呼吸频率",
    "干体重",
    "透析液钙浓度",
    "透析液电导率",
    "抗凝剂类型",
    "透析后体重",
    "实际透析时长",
    "瘘管类型",
    "瘘管位置",
    "瘘管使用时间",
    "首次透析年龄",
    "透中高血压_计算",
    "透中低血压_计算",
    "涨幅时间点比值",
    "涨幅时间点比值区间",
    "涨幅时间点差值",
    "涨幅时间点差值区间",
    "降幅时间点比值区间",
    "降幅时间点差值区间",
    "透析中收缩压_mean",
    "透析中收缩压_std",
    "透析中舒张压_mean",
    "透析中舒张压_std",
    "透析中脉搏_mean",
    "透析中脉搏_std",
    "超滤率_mean",
    "超滤率_std",
    "超滤量MAX",
    "静脉压_mean",
    "静脉压_std",
    "动脉压_mean",
    "动脉压_std",
    "血流速_mean",
    "血流速_std",
    "透析液温度_mean",
    "透析液温度_std",
    "跨膜压_mean",
    "跨膜压_std",
    "历史平均透前体重",
    "历史平均透前呼吸频率",
    "历史平均透前体温",
    "历史平均干体重",
    "历史平均透析液钙浓度",
    "历史平均透析液电导率",
    "历史平均实际透析时长",
    "历史平均透前收缩压",
    "历史平均透前舒张压",
    "历史平均涨幅时间点比值区间",
    "历史平均涨幅时间点差值区间",
    "历史平均透中高血压_计算",
    "历史平均透析中收缩压_mean",
    "历史平均透析中舒张压_mean",
    "历史平均透析中脉搏_mean",
    "历史平均超滤率_mean",
    "历史平均静脉压_mean",
    "历史平均动脉压_mean",
    "历史平均血流速_mean",
    "历史平均透析液温度_mean",
    "历史平均跨膜压_mean",
]
