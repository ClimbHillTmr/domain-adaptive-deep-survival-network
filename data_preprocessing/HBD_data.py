import os
import pandas as pd
import numpy as np
from data_process import (
    replace_extremes_with_percentiles,
    calculate_pressure_change,
    calculate_time_points,
    convert_to_float_list,
    diagnose_hypertension,
)
import ast
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading


def process_patient_historical_averages(patient_data, mean_columns):
    """处理单个患者的历史平均值计算"""
    patient_id = patient_data['患者id'].iloc[0]
    patient_data = patient_data.sort_values(by="透析日期")
    
    # 创建结果字典
    result_data = {
        '患者id': [],
        '透析日期': []
    }
    
    # 初始化所有历史平均列
    for col in mean_columns:
        result_data["历史平均" + col] = []
    
    # 遍历该患者的每一行
    for index, row in patient_data.iterrows():
        dialysis_date = row["透析日期"]
        
        # 在透析日期之前，筛选出历史记录
        historical_records = patient_data[patient_data["透析日期"] < dialysis_date]
        
        # 添加患者id和透析日期
        result_data['患者id'].append(patient_id)
        result_data['透析日期'].append(dialysis_date)
        
        # 计算每个指标的历史平均值
        for col in mean_columns:
            if historical_records.shape[0] > 0:
                avg_value = historical_records[col].mean()
            else:
                avg_value = 0
            result_data["历史平均" + col].append(avg_value)
    
    return pd.DataFrame(result_data)


def process_historical_averages_shenyi_parallel(dataset, first_pressure_sd="", n_threads=20):
    """使用多线程处理深医历史平均值计算的函数"""
    mean_columns = [
        "透前体重",
        "透前呼吸频率",
        "透前体温",
        "干体重",
        "透析液钙浓度",
        "透析液电导率",
        "实际透析时长",
        "透前收缩压",
        "透前舒张压",
        "涨幅时间点比值区间",
        "涨幅时间点差值区间",
        "降幅时间点比值",
        "降幅时间点差值",
        "涨幅时间点比值",
        "涨幅时间点差值",
        "降幅时间点比值区间",
        "降幅时间点差值区间",
        "透中高血压_计算",
        "透中低血压_计算",
        "透前体重-干体重",
        "透析中收缩压_mean",
        "透析中舒张压_mean",
        "透析中脉搏_mean",
        "超滤率_mean",
        "静脉压_mean",
        "动脉压_mean",
        "血流速_mean",
        "透析液温度_mean",
        "跨膜压_mean",
        "透前动脉压",
        "透后动脉压",
        "超滤量MAX",
    ]
    
    dataset = dataset.sort_values(by=["患者id", "透析日期"])
    optimized_file_path = f"/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_result_df.csv"
    if os.path.exists(optimized_file_path):
        print(f"发现优化数据文件: {optimized_file_path}")
        print("跳过中间计算步骤")
        result_df = pd.read_csv(optimized_file_path)
    else:
        print(f"开始使用 {n_threads} 个线程处理历史平均值计算...")
        
        # 按患者分组
        patient_groups = [group for _, group in dataset.groupby('患者id')]
        total_patients = len(patient_groups)
        print(f"总共需要处理 {total_patients} 个患者的数据")
        
        # 使用线程池处理
        result_dfs = []
        completed_patients = 0
        lock = threading.Lock()
        
        def process_with_progress(patient_data):
            nonlocal completed_patients
            result = process_patient_historical_averages(patient_data, mean_columns)
            
            with lock:
                completed_patients += 1
                if completed_patients % 100 == 0 or completed_patients == total_patients:
                    progress = completed_patients / total_patients * 100
                    print(f"历史平均值计算进度: {completed_patients}/{total_patients} ({progress:.1f}%)")
            
            return result
        
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            # 提交所有任务
            future_to_patient = {executor.submit(process_with_progress, patient_data): i 
                            for i, patient_data in enumerate(patient_groups)}
            
            # 收集结果
            for future in as_completed(future_to_patient):
                try:
                    result_df = future.result()
                    result_dfs.append(result_df)
                except Exception as exc:
                    patient_idx = future_to_patient[future]
                    print(f'患者 {patient_idx} 处理时发生异常: {exc}')
        
        # 合并所有结果
        print("合并处理结果...")
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        print("保存历史平均值结果...")
        result_df.to_csv(f"/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_result_df{first_pressure_sd}.csv")
    
    # 合并原始数据和历史平均值数据
    print("合并原始数据和历史平均值数据...")
    whole_data = pd.merge(
        dataset.reset_index(), result_df, on=["患者id", "透析日期"], how="inner"
    )
    
    # 重置索引以避免重复标签错误
    whole_data = whole_data.reset_index(drop=True)
    
    # 计算累积平均值（这部分保持原有逻辑）
    print("计算累积平均值...")
    for col in mean_columns:
        whole_data['历史平均' + col] = whole_data.groupby('患者id')[col].cumsum() / (
            whole_data.groupby('患者id').cumcount() + 1
        )
    
    # 计算历史比率（使用多线程优化）
    print("开始计算历史比率...")
    Y_rate = process_historical_rates_parallel(dataset, n_threads)
    
    # 合并最终数据
    final_data = pd.concat([whole_data, Y_rate], axis=1)
    
    print("保存最终数据...")
    final_data.to_csv(f"/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_final_data{first_pressure_sd}.csv")
    
    return dataset


def process_patient_historical_rates(patient_data):
    """处理单个患者的历史比率计算"""
    patient_data = patient_data.sort_values(by="透析日期")
    
    history_HBP = 0
    history_LBP_times_0 = 0
    history_LBP_times_1 = 0
    history_LBP_times_2 = 0
    history_LBP_times_3 = 0
    history_LBP_times_4 = 0
    total_times = 1
    
    patient_rates = []
    
    for i, row in patient_data.iterrows():
        if total_times == 1:
            # 第一次透析，历史比率为0
            y_rate = [0, 0, 0, 0, 0, 0]
        else:
            # 计算历史比率
            history_HBP_rate_rate = history_HBP / total_times
            history_HBP_time_rate_0_rate = history_LBP_times_0 / total_times
            history_HBP_time_rate_1_rate = history_LBP_times_1 / total_times
            history_HBP_time_rate_2_rate = history_LBP_times_2 / total_times
            history_HBP_time_rate_3_rate = history_LBP_times_3 / total_times
            history_HBP_time_rate_4_rate = history_LBP_times_4 / total_times
            
            y_rate = [
                history_HBP_rate_rate,
                history_HBP_time_rate_0_rate,
                history_HBP_time_rate_1_rate,
                history_HBP_time_rate_2_rate,
                history_HBP_time_rate_3_rate,
                history_HBP_time_rate_4_rate,
            ]
        
        patient_rates.append(y_rate)
        
        # 更新计数器
        if row["透中低血压_计算"] == 0:
            history_LBP_times_0 += 1
        if row["透中低血压_计算"] == 1:
            history_HBP += 1
        if row["降幅时间点比值区间"] == 1:
            history_LBP_times_1 += 1
        if row["降幅时间点比值区间"] == 2:
            history_LBP_times_2 += 1
        if row["降幅时间点比值区间"] == 3:
            history_LBP_times_3 += 1
        if row["降幅时间点比值区间"] == 4:
            history_LBP_times_4 += 1
        
        total_times += 1
    
    return patient_rates


def process_historical_rates_parallel(dataset, n_threads=20):
    """使用多线程处理历史比率计算"""
    dataset = dataset.sort_values(by=["患者id", "透析日期"])
    
    # 按患者分组
    patient_groups = [group for _, group in dataset.groupby('患者id')]
    total_patients = len(patient_groups)
    print(f"使用 {n_threads} 个线程处理 {total_patients} 个患者的历史比率计算...")
    
    all_rates = []
    completed_patients = 0
    lock = threading.Lock()
    
    def process_with_progress(patient_data):
        nonlocal completed_patients
        rates = process_patient_historical_rates(patient_data)
        
        with lock:
            completed_patients += 1
            if completed_patients % 100 == 0 or completed_patients == total_patients:
                progress = completed_patients / total_patients * 100
                print(f"历史比率计算进度: {completed_patients}/{total_patients} ({progress:.1f}%)")
        
        return rates
    
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # 提交所有任务
        future_to_patient = {executor.submit(process_with_progress, patient_data): i 
                           for i, patient_data in enumerate(patient_groups)}
        
        # 收集结果
        for future in as_completed(future_to_patient):
            try:
                patient_rates = future.result()
                all_rates.extend(patient_rates)
            except Exception as exc:
                patient_idx = future_to_patient[future]
                print(f'患者 {patient_idx} 历史比率计算时发生异常: {exc}')
    
    # 转换为DataFrame
    Y_rate = pd.DataFrame(
        all_rates,
        columns=[
            'history_HBP',
            'history_LBP_times_0',
            'history_LBP_times_1',
            'history_LBP_times_2',
            'history_LBP_times_3',
            'history_LBP_times_4',
        ],
    )
    
    return Y_rate


def process_historical_averages_shenyi(dataset, first_pressure_sd=""):
    """处理深医历史平均值计算的函数（原始版本，保留作为备份）"""
    mean_columns = [
        # "患者状态",
        # "透析前预设UFV",
        "透前体重",
        "透前呼吸频率",
        "透前体温",
        "干体重",
        "透析液钙浓度",
        "透析液电导率",
        # "透析液钠浓度",
        # "透析液钾浓度",
        # "透析液碳酸氢根浓度",
        "实际透析时长",
        "透前收缩压",
        "透前舒张压",
        "涨幅时间点比值区间",
        "涨幅时间点差值区间",
        "降幅时间点比值",
        "降幅时间点差值",
        "涨幅时间点比值",
        "涨幅时间点差值",
        "降幅时间点比值区间",
        "降幅时间点差值区间",
        "透中高血压_计算",
        "透中低血压_计算",
        "透前体重-干体重",
        "透析中收缩压_mean",
        "透析中舒张压_mean",
        "透析中脉搏_mean",
        # "平均动脉压_mean",
        "超滤率_mean",
        # "超滤量_mean",
        "静脉压_mean",
        "动脉压_mean",
        "血流速_mean",
        "透析液温度_mean",
        "跨膜压_mean",
        "透前动脉压",
        "透后动脉压",
        "超滤量MAX",
    ]
    dataset = dataset.sort_values(by=["患者id", "透析日期"])

    # 创建一个新的DataFrame来存储结果
    result_df = pd.DataFrame()

    # 遍历mean_columns中的每一列
    for col in mean_columns:
        # 创建空列表来存储历史平均指标、透析日期和患者ID
        historical_avg_values = []
        dialysis_dates = []
        patient_ids = []

        # 遍历DataFrame中的每一行
        for index, row in dataset.iterrows():
            # 获取当前行的患者id和透析日期
            patient_id = row["患者id"]
            dialysis_date = row["透析日期"]

            # 在透析日期之前，筛选出同一患者的历史记录
            historical_records = dataset[
                (dataset["患者id"] == patient_id)
                & (dataset["透析日期"] < dialysis_date)
            ]

            # 计算历史记录中治疗指标的平均值
            if historical_records.shape[0] > 0:
                avg_value = historical_records[col].mean()
            else:
                avg_value = 0

            # 将患者id、透析日期和平均值添加到列表中
            patient_ids.append(patient_id)
            dialysis_dates.append(dialysis_date)
            historical_avg_values.append(avg_value)

        # 将患者id、透析日期和历史平均值列表添加到结果DataFrame中
        result_df["患者id"] = patient_ids
        result_df["透析日期"] = dialysis_dates
        result_df["历史平均" + col] = historical_avg_values

    print(result_df)
    result_df.to_csv("/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_result_df" + str(first_pressure_sd) + ".csv")

    whole_data = pd.merge(
        dataset.reset_index(), result_df, on=["患者id", "透析日期"], how="inner"
    )

    # whole_data = pd.concat([dataset.reset_index(), result_df], axis=1)

    for col in mean_columns:
        whole_data['历史平均' + col] = whole_data.groupby('患者id')[col].cumsum() / (
            whole_data.groupby('患者id').cumcount() + 1
        )

    Y_rate = pd.DataFrame(
        columns=[
            'history_HBP',
            'history_LBP_times_0',
            'history_LBP_times_1',
            'history_LBP_times_2',
            'history_LBP_times_3',
            'history_LBP_times_4',
        ]
    )

    Y_rate = []

    # 计算平均值
    # 计算历史比率
    history_HBP = 0
    history_LBP_times_0 = 0
    history_LBP_times_1 = 0
    history_LBP_times_2 = 0
    history_LBP_times_3 = 0
    history_LBP_times_4 = 0
    total_times = 1
    last_p = ""
    dataset = dataset.sort_values(by=["患者id", "透析日期"])

    for i in range(len(dataset)):
        # for i in range(0, 1000):
        if i % 1000 == 0:
            print("加载数据", f"{str(round(i / len(dataset) * 100, 10))}%")

        p = dataset["患者id"].iloc[i]

        if last_p == p:
            history_HBP_rate_rate = history_HBP / total_times
            history_HBP_time_rate_0_rate = history_LBP_times_0 / total_times
            history_HBP_time_rate_1_rate = history_LBP_times_1 / total_times
            history_HBP_time_rate_2_rate = history_LBP_times_2 / total_times
            history_HBP_time_rate_3_rate = history_LBP_times_3 / total_times
            history_HBP_time_rate_4_rate = history_LBP_times_4 / total_times

            y_rate = [
                history_HBP_rate_rate,
                history_HBP_time_rate_0_rate,
                history_HBP_time_rate_1_rate,
                history_HBP_time_rate_2_rate,
                history_HBP_time_rate_3_rate,
                history_HBP_time_rate_4_rate,
            ]

            if dataset["透中低血压_计算"].iloc[i] == 0:
                history_HBP = 0
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 4:
                history_LBP_times_4 += 1
            total_times += 1
        else:
            last_p = p
            total_times = 1

            history_HBP = 0
            history_LBP_times_0 = 0
            history_LBP_times_1 = 0
            history_LBP_times_2 = 0
            history_LBP_times_3 = 0
            history_LBP_times_4 = 0

            y_rate = [0, 0, 0, 0, 0, 0]

            if dataset["透中低血压_计算"].iloc[i] == 0:
                history_HBP = 0
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点比值区间"].iloc[i] == 4:
                history_LBP_times_4 += 1

        Y_rate.append(y_rate)

    Y_rate = pd.DataFrame(
        Y_rate,
        columns=[
            'history_HBP',
            'history_LBP_times_0',
            'history_LBP_times_1',
            'history_LBP_times_2',
            'history_LBP_times_3',
            'history_LBP_times_4',
        ],
    )

    final_data = pd.concat([whole_data, Y_rate], axis=1)

    final_data.to_csv("/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_final_data" + str(first_pressure_sd) + ".csv")
    return dataset


def shengyi_dataset(first_pressure_sd="", use_parallel=True, n_threads=20):
    """主函数，支持选择是否使用并行处理"""
    # 检查优化数据文件是否存在
    optimized_file_path = "/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_optimized_data.csv"
    if os.path.exists(optimized_file_path):
        print(f"发现优化数据文件: {optimized_file_path}")
        print("跳过中间计算步骤，直接进行历史平均值计算...")
        dataset = pd.read_csv(optimized_file_path)
        
        if use_parallel:
            print(f"使用并行处理模式，线程数: {n_threads}")
            return process_historical_averages_shenyi_parallel(dataset, first_pressure_sd, n_threads)
        else:
            print("使用串行处理模式")
            return process_historical_averages_shenyi(dataset, first_pressure_sd)
    
    # Read the dataset
    dataset = pd.read_csv("/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/updated_dataset_shenyi.csv")

    # 解析字符串为列表
    def safe_literal_eval(x):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return []

    dataset["透中不良反应"] = dataset["透中不良反应"].apply(safe_literal_eval)
    dataset["透中不良反应时间"] = dataset["透中不良反应时间"].apply(safe_literal_eval)

    # 统计所有不良反应类型
    all_reactions = set()
    for reactions in dataset["透中不良反应"]:
        all_reactions.update(reactions)

    # 为每一种不良反应建立新列
    for reaction in all_reactions:
        dataset[reaction] = dataset["透中不良反应"].apply(
            lambda x: 1 if reaction in x else 0
        )
        dataset[reaction + "_时间"] = dataset.apply(
            lambda row: (
                row["透中不良反应_time"][row["透中不良反应"].index(reaction)]
                if reaction in row["透中不良反应"] and row["透中不良反应时间"]
                else None
            ),
            axis=1,
        )

    dataset["透前动脉压"] = (
        1 / 3 * dataset["透前收缩压"] + 2 / 3 * dataset["透前舒张压"]
    )
    dataset["透后动脉压"] = (
        1 / 3 * dataset["透后收缩压"] + 2 / 3 * dataset["透后舒张压"]
    )

    # Data filtering
    dataset = dataset[
        dataset["透析中收缩压"].notnull() & (dataset["透析中收缩压"] != "")
    ]
    # dataset = dataset[(dataset["干体重"] > 0) & (dataset["实际透析时长"] > 0)]

    dataset.rename(columns={"动脉压": "原始动脉压"}, inplace=True)

    # 处理列表数据
    list_columns = [
        "透析中收缩压",
        "透析中舒张压",
        "透析中脉搏",
        # "平均动脉压",
        "超滤率",
        "超滤量",
        "静脉压",
        "原始动脉压",
        "血流速",
        "透析液温度",
        "跨膜压",
    ]

    for col in list_columns:
        print(col)
        # dataset[col] = dataset[col].str.strip("[]").str.split(",")
        dataset[col] = dataset[col].apply(convert_to_float_list)

    dataset["透析中收缩压"] = dataset["透析中收缩压"].apply(
        lambda lst: lst if lst else []
    )
    dataset = dataset[dataset["透析中收缩压"].apply(len) > 0]
    dataset["透析中舒张压"] = dataset["透析中舒张压"].apply(
        lambda lst: lst if lst else []
    )
    dataset = dataset[dataset["透析中舒张压"].apply(len) > 0]

    # # 透析液问题：深医39度，福鼎37度，处理方案：低于30、高于40予以去除。
    dataset["透析液温度"] = dataset["透析液温度"].apply(
        lambda lst: [] if lst is None else [i for i in lst if 30 < i < 40]
    )

    # 计算动脉压并添加到 DataFrame 中
    dataset["动脉压"] = dataset.apply(
        lambda row: [
            (1 / 3) * sbp + (2 / 3) * dbp
            for sbp, dbp in zip(row["透析中收缩压"], row["透析中舒张压"])
        ],
        axis=1,
    )
    list_columns = list_columns + ["动脉压"]
    dataset = dataset[dataset["动脉压"].notna()]
    # dataset = dataset[
    #     (dataset["干体重"] > 0) & dataset["动脉压"].notna() & (dataset["实际透析时长"] > 0)
    # ]

    # 替代方法：使用 lambda 表达式计算最大值
    dataset["超滤量MAX"] = dataset["超滤量"].apply(
        lambda lst: np.nan if lst is None or len(lst) == 0 else np.max(lst)
    )

    # 使用函数替换极值
    dataset = replace_extremes_with_percentiles(dataset, list_columns)

    # Remove rows with empty lists
    dataset = dataset[dataset["透析中收缩压"].apply(lambda lst: lst and len(lst) > 0)]
    dataset["透中数据记录时间节点"] = (
        dataset["透中数据记录时间节点"].str.strip("[]").str.split(",")
    )

    dataset = dataset[dataset["动脉压"].apply(lambda lst: len(lst) > 0)]

    # Calculate derived features
    dataset["透析年龄"] = (
        pd.to_datetime(dataset["透析日期"]).dt.year
        - pd.to_datetime(dataset["出生日期"]).dt.year
    )
    dataset["首次透析年龄"] = (
        pd.to_datetime(dataset["首次透析日期"]).dt.year
        - pd.to_datetime(dataset["出生日期"]).dt.year
    )
    dataset["瘘管使用时间"] = (
        pd.to_datetime(dataset["透析日期"]).dt.year
        - pd.to_datetime(dataset["瘘管置管时间"]).dt.year
    )

    dataset["高血压诊断"] = dataset["诊断"].apply(diagnose_hypertension)

    # Encode gender using LabelEncoder
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    dataset["性别"] = label_encoder.fit_transform(dataset["性别"])

    dataset["透中高血压_计算"] = dataset.apply(
        lambda row: calculate_pressure_change(
            row=row, first_pressure=first_pressure_sd, pressure_type="hypertension"
        ),
        axis=1,
    )
    dataset["透中高血压_计算"].fillna(0, inplace=True)
    dataset["透中低血压_计算"] = dataset.apply(
        lambda row: calculate_pressure_change(
            row=row, first_pressure=first_pressure_sd, pressure_type="hypotension"
        ),
        axis=1,
    )
    dataset["透中低血压_计算"].fillna(0, inplace=True)

    # Remove empty strings from '透中数据记录时间节点' column
    dataset["透中数据记录时间节点"] = dataset["透中数据记录时间节点"].apply(
        lambda lst: [item for item in lst if item != ""]
    )

    dataset["涨幅时间点"] = dataset.apply(
        lambda row: calculate_time_points(
            row=row, first_pressure=first_pressure_sd, pressure_type="hypertension"
        ),
        axis=1,
    )
    dataset["降幅时间点"] = dataset.apply(
        lambda row: calculate_time_points(
            row=row, first_pressure=first_pressure_sd, pressure_type="hypotension"
        ),
        axis=1,
    )

    # Extract '透析开始时间' and '透析结束时间' from '透中数据记录时间节点'
    dataset["透析开始时间"] = dataset["透中数据记录时间节点"].apply(lambda lst: lst[0])
    dataset["透析结束时间"] = dataset["透中数据记录时间节点"].apply(lambda lst: lst[-1])

    # Convert time strings to datetime objects
    dataset["透析开始时间"] = pd.to_datetime(dataset["透析开始时间"])
    dataset["透析结束时间"] = pd.to_datetime(dataset["透析结束时间"])
    
    # Convert 涨幅时间点 and 降幅时间点 to datetime, handling None values
    dataset["涨幅时间点"] = pd.to_datetime(dataset["涨幅时间点"], errors='coerce')
    dataset["降幅时间点"] = pd.to_datetime(dataset["降幅时间点"], errors='coerce')

    # Calculate '涨幅时间点区间' and '涨幅时间点差值' columns
    time_diff_minutes = (
        dataset["透析结束时间"] - dataset["透析开始时间"]
    ).dt.total_seconds()
    
    # Handle cases where 涨幅时间点 is NaT (Not a Time)
    valid_涨幅时间点 = dataset["涨幅时间点"].notna()
    dataset["涨幅时间点比值"] = np.nan
    dataset.loc[valid_涨幅时间点, "涨幅时间点比值"] = (
        dataset.loc[valid_涨幅时间点, "涨幅时间点"] - dataset.loc[valid_涨幅时间点, "透析开始时间"]
    ).dt.total_seconds() / time_diff_minutes.loc[valid_涨幅时间点]
    dataset["涨幅时间点比值区间"] = dataset["涨幅时间点比值"].apply(
        lambda x: (
            0
            if x < 0
            else (1 if x <= 0.25 else (2 if x <= 0.5 else (3 if x <= 0.75 else 4)))
        )
    )
    # Calculate 涨幅时间点差值 only for valid 涨幅时间点
    dataset["涨幅时间点差值"] = np.nan
    dataset.loc[valid_涨幅时间点, "涨幅时间点差值"] = (
        (dataset.loc[valid_涨幅时间点, "涨幅时间点"] - dataset.loc[valid_涨幅时间点, "透析开始时间"]).dt.total_seconds() / 60 / 60
    )
    dataset["涨幅时间点差值区间"] = dataset["涨幅时间点差值"].apply(
        lambda x: (
            0 if x < 0 else (1 if x <= 1 else (2 if x <= 2 else (3 if x <= 3 else 4)))
        )
    )
    # Calculate '降幅时间点比值' and '降幅时间点差值' columns
    # Handle cases where 降幅时间点 is NaT (Not a Time)
    valid_降幅时间点 = dataset["降幅时间点"].notna()
    dataset["降幅时间点比值"] = np.nan
    dataset.loc[valid_降幅时间点, "降幅时间点比值"] = (
        dataset.loc[valid_降幅时间点, "降幅时间点"] - dataset.loc[valid_降幅时间点, "透析开始时间"]
    ).dt.total_seconds() / time_diff_minutes.loc[valid_降幅时间点]
    dataset["降幅时间点比值区间"] = dataset["降幅时间点比值"].apply(
        lambda x: (
            0
            if x < 0
            else (1 if x <= 0.25 else (2 if x <= 0.5 else (3 if x <= 0.75 else 4)))
        )
    )
    # Calculate 降幅时间点差值 only for valid 降幅时间点
    dataset["降幅时间点差值"] = np.nan
    dataset.loc[valid_降幅时间点, "降幅时间点差值"] = (
        (dataset.loc[valid_降幅时间点, "降幅时间点"] - dataset.loc[valid_降幅时间点, "透析开始时间"]).dt.total_seconds() / 60 / 60
    )
    dataset["降幅时间点差值区间"] = dataset["降幅时间点差值"].apply(
        lambda x: (
            0 if x < 0 else (1 if x <= 1 else (2 if x <= 2 else (3 if x <= 3 else 4)))
        )
    )
    dataset["透前体重-干体重"] = dataset["透前体重"] - dataset["干体重"]

    # Process mean and standard deviation of list columns
    for col in list_columns:
        mean_col = col + "_mean"
        std_col = col + "_std"
        dataset[mean_col] = dataset[col].apply(
            lambda lst: np.mean(lst) if lst else None
        )
        dataset[std_col] = dataset[col].apply(lambda lst: np.std(lst) if lst else None)

    keep_columns = [
        "患者id",
        "透析记录id",
        "年龄",
        "性别",
        "透析龄",
        "出生日期",
        "传染病",
        # "患者状态",
        "透前收缩压",
        "透前舒张压",
        "透前动脉压",
        "透后动脉压",
        "透后收缩压",
        "透后舒张压",
        "透析龄_天数",
        "透析年龄",
        "透析日期",
        # "透析器",
        "透析方式",
        # "透析前预设UFV",
        "透前体重",
        "透前呼吸频率",
        "透前体温",
        "干体重",
        "透析液钙浓度",
        "透析液电导率",
        # "透析液钠浓度",
        # "透析液钾浓度",
        # "透析液碳酸氢根浓度",
        "抗凝剂类型",
        # "抗凝剂使用总量",
        # "抗凝剂维持量",
        # "抗凝剂追加量",
        "透析后体重",
        # "透后脉搏",
        # "透后体温",
        "实际透析时长",
        "瘘管类型",
        "瘘管位置",
        # '瘘管置管时间',
        "瘘管使用时间",
        "首次透析年龄",
        "高血压诊断",
        "透中高血压_计算",
        "透中低血压_计算",
        "涨幅时间点",
        "透析开始时间",
        "透析结束时间",
        "涨幅时间点比值",
        "涨幅时间点比值区间",
        '涨幅时间点差值区间',
        "降幅时间点",
        "降幅时间点比值",
        "降幅时间点比值区间",
        "透前体重-干体重",
        "涨幅时间点差值",
        "降幅时间点差值",
        "降幅时间点差值区间",
        "降幅时间点比值区间",
        "超滤量MAX",
        "透析中收缩压_mean",
        "透析中收缩压_std",
        "透析中舒张压_mean",
        "透析中舒张压_std",
        "透析中脉搏_mean",
        "透析中脉搏_std",
        # "平均动脉压_mean",
        # "平均动脉压_std",
        "超滤率_mean",
        "超滤率_std",
        # "超滤量_mean",
        # "超滤量_std",
        "静脉压_mean",
        "静脉压_std",
        "动脉压_mean",
        "动脉压_std",
        "原始动脉压_mean",
        "原始动脉压_std",
        "血流速_mean",
        "血流速_std",
        "透析液温度_mean",
        "透析液温度_std",
        "跨膜压_mean",
        "跨膜压_std",
    ]

    # Save the resulting DataFrame
    dataset = dataset[keep_columns]
    dataset.to_csv("/home/cht/Works/PredictionTimeHypotensionDialysis/data_preprocessing/data/深医_optimized_data" + str(first_pressure_sd) + ".csv")
    # dataset.to_csv("Final_data/深医_optimized_data"+str(first_pressure_sd)".csv")
    # del X['涨幅时间点区间']
    # del X['透中高血压_计算']
    
    # 调用历史平均值处理函数
    if use_parallel:
        print(f"使用并行处理模式，线程数: {n_threads}")
        return process_historical_averages_shenyi_parallel(dataset, first_pressure_sd, n_threads)
    else:
        print("使用串行处理模式")
        return process_historical_averages_shenyi(dataset, first_pressure_sd)


# 主执行部分 - 使用20个线程进行并行计算
I=""
data = shengyi_dataset(i, use_parallel=True, n_threads=30)
print(f"=== {i} 数据处理完成 ===")
