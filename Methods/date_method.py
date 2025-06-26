import pandas as pd
from datetime import datetime

def split_dataset_by_date(df, date_column, train_ratio=0.8):
    """
    分割数据集为训练集和测试集，并返回包含训练集、测试集以及日期起止点的字典。

    Parameters:
        df (pd.DataFrame): 包含数据的DataFrame。
        date_column (str): 日期时间列的列名。
        train_ratio (float): 用于训练集的比例，默认为0.8。

    Returns:
        tuple: 返回一个包含训练集、测试集以及日期起止点的字典的元组。
    """
    # 将日期列转换为日期时间类型，指定日期的格式
    # df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y')

    # 根据日期进行排序
    df.sort_values(by=date_column, inplace=True)

    # 计算要用于训练集的数据量
    train_size = int(train_ratio * len(df))

    # 分割数据集为训练集和测试集
    train_set = df.iloc[:train_size]
    test_set = df.iloc[train_size:]

    # 获取训练集和测试集的日期起止点
    train_start_date = train_set[date_column].min()
    train_end_date = train_set[date_column].max()
    test_start_date = test_set[date_column].min()
    test_end_date = test_set[date_column].max()

    # 创建一个包含日期起止点的字典
    date_ranges = {
        '训练集起始日期': train_start_date,
        '训练集结束日期': train_end_date,
        '测试集起始日期': test_start_date,
        '测试集结束日期': test_end_date
    }

    # 返回训练集、测试集和日期起止点的字典
    return train_set, test_set, date_ranges

# # 示例数据（日期格式为day/month/year）
# data = {
#     '透析日期': ['01/01/2023', '02/01/2023', '03/01/2023', '04/01/2023', '05/01/2023', '06/01/2023'],
#     '其他列': [1, 2, 3, 4, 5, 6]
# }

# df = pd.DataFrame(data)

# # 使用函数分割数据集并获取日期起止点、训练集和测试集
# train_set, test_set, date_ranges = split_dataset_by_date(df, '透析日期')

# # 输出结果
# print("训练集:")
# print(train_set)
# print("\n测试集:")
# print(test_set)
# print("\n日期起止点字典:")
# print(date_ranges)
