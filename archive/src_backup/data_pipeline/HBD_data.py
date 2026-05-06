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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import threading

dataset = pd.read_csv("/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/updated_dataset_shenyi.csv")
dataset["患者id"].nunique()

def shengyi_dataset(first_pressure_sd="透前动脉压", use_parallel=True, n_threads=30):
    """主函数，支持选择是否使用并行处理"""
    # 检查优化数据文件是否存在
    optimized_file_path = "/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/data/深医_optimized_data透前动脉压.csv"
    if os.path.exists(optimized_file_path):
        print(f"发现优化数据文件: {optimized_file_path}")
        print("跳过中间计算步骤，直接进行历史平均值计算...")
        dataset = pd.read_csv(optimized_file_path)

        if use_parallel:
            print(f"使用并行处理模式，线程数: {n_threads}")
            return process_historical_averages(dataset,first_pressure_sd="透前动脉压", use_parallel=use_parallel, n_threads=30)
        else:
            print("使用串行处理模式")
            return process_historical_averages_shenyi(dataset, first_pressure_sd)
    else:
        # Read the dataset
        dataset = pd.read_csv("/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/updated_dataset_shenyi.csv")

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

    # 处理透前动脉压的极值
    q5_pre = dataset['透前动脉压'].quantile(0.05)
    q95_pre = dataset['透前动脉压'].quantile(0.95)
    dataset['透前动脉压'] = dataset['透前动脉压'].apply(
        lambda x: q5_pre if x < q5_pre else (q95_pre if x > q95_pre else x)
    )

    # 处理透前动脉压的极值
    q5_pre = dataset['透前收缩压'].quantile(0.05)
    q95_pre = dataset['透前收缩压'].quantile(0.95)
    dataset['透前收缩压'] = dataset['透前收缩压'].apply(
        lambda x: q5_pre if x < q5_pre else (q95_pre if x > q95_pre else x)
    )

    # 处理列表数据
    list_columns = [
        "透析中收缩压",
        "透析中舒张压",
        "透析中脉搏",
        # "平均动脉压",
        # "动脉压",
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

    # dataset = dataset[dataset["动脉压"].apply(lambda lst: len(lst) > 0)]
    # dataset = dataset[
    #     (dataset["干体重"] > 0) & dataset["动脉压"].notna() & (dataset["实际透析时长"] > 0)
    # ]

    # 替代方法：使用 lambda 表达式计算最大值
    dataset["超滤量MAX"] = dataset["超滤量"].apply(
        lambda lst: np.nan if lst is None or len(lst) == 0 else np.max(lst)
    )

    # 修复超滤量MAX异常值：使用99%分位数截断
    q99_uf = dataset["超滤量MAX"].quantile(0.99)
    dataset["超滤量MAX"] = dataset["超滤量MAX"].clip(upper=q99_uf)
    print(f"超滤量MAX 99%分位数截断值: {q99_uf:.2f}")

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
    dataset["透中高血压_计算"] = dataset["透中高血压_计算"].fillna(0)
    dataset["透中低血压_计算"] = dataset.apply(
        lambda row: calculate_pressure_change(
        row=row, first_pressure='透前收缩压', pressure_type="hypotension"
        # lambda row: calculate_pressure_change(
        #     row=row, first_pressure=first_pressure_sd, pressure_type="hypotension"
        ),
        axis=1,
    )
    dataset["透中低血压_计算"] = dataset["透中低血压_计算"].fillna(0)

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
            row=row, first_pressure='透前收缩压', pressure_type="hypotension"
        ),
        axis=1,
    )

    # Extract '透析开始时间' and '透析结束时间' from '透中数据记录时间节点'
    dataset["透析开始时间"] = dataset["透中数据记录时间节点"].apply(lambda lst: lst[0])
    dataset["透析结束时间"] = dataset["透中数据记录时间节点"].apply(lambda lst: lst[-1])

    dataset["涨幅时间点"] = pd.to_datetime(dataset["涨幅时间点"])
    dataset["降幅时间点"] = pd.to_datetime(dataset["降幅时间点"])
    dataset["透析开始时间"] = pd.to_datetime(dataset["透析开始时间"])
    dataset["透析结束时间"] = pd.to_datetime(dataset["透析结束时间"])

    # 相对起始分钟数组与时长
    def minutes_from_start_list(lst):
        try:
            base = pd.to_datetime(lst[0])
            return [float((pd.to_datetime(t) - base).total_seconds() / 60.0) for t in lst]
        except Exception:
            return []
    dataset["minutes_from_start_list"] = dataset["透中数据记录时间节点"].apply(minutes_from_start_list)
    dataset["duration_minutes"] = (dataset["透析结束时间"] - dataset["透析开始时间"]).dt.total_seconds() / 60.0

    # 事件分钟 et_min 与事件标记 events
    dataset["et_min"] = (
        (dataset["降幅时间点"] - dataset["透析开始时间"]).dt.total_seconds() / 60.0
    )
    dataset["events"] = dataset["透中低血压_计算"].astype(int)
    dataset.loc[dataset["events"] == 0, "et_min"] = dataset.loc[dataset["events"] == 0, "duration_minutes"]

    # 阶段标签（两套边界）
    def stage_label_from_et(et, b1, b2, b3):
        if et is None:
            return 0
        try:
            x = float(et)
        except Exception:
            return 0
        return 1 if x <= b1 else (2 if x <= b2 else (3 if x <= b3 else 3))
    dataset["stage_30_60_120"] = dataset["et_min"].apply(lambda x: stage_label_from_et(x, 30, 60, 120))
    dataset["stage_30_90_120"] = dataset["et_min"].apply(lambda x: stage_label_from_et(x, 30, 90, 120))

    # Calculate '涨幅时间点区间' and '涨幅时间点差值' columns
    time_diff_minutes = (
        dataset["透析结束时间"] - dataset["透析开始时间"]
    ).dt.total_seconds()
    dataset["涨幅时间点比值"] = (
        dataset["涨幅时间点"] - dataset["透析开始时间"]
    ).dt.total_seconds() / time_diff_minutes
    dataset["涨幅时间点比值区间"] = dataset["涨幅时间点比值"].apply(
        lambda x: (
            0
            if x <= 0
            else (1 if x <= 0.25 else (2 if x <= 0.5 else (3 if x <= 0.75 else 4)))
        )
    )
    dataset["涨幅时间点差值"] = (
        (dataset["涨幅时间点"] - dataset["透析开始时间"]).dt.total_seconds() / 60 / 60
    )
    dataset["涨幅时间点差值区间"] = dataset["涨幅时间点差值"].apply(
        lambda x: (
            0 if x <= 0 else (1 if x <= 1 else (2 if x <= 2 else (3 if x <= 3 else 4)))
        )
    )
    # Calculate '降幅时间点比值' and '降幅时间点差值' columns
    dataset["降幅时间点比值"] = (
        dataset["降幅时间点"] - dataset["透析开始时间"]
    ).dt.total_seconds() / time_diff_minutes
    dataset["降幅时间点比值区间"] = dataset["降幅时间点比值"].apply(
        lambda x: (
            0
            if x <= 0
            else (1 if x <= 0.25 else (2 if x <= 0.5 else (3 if x <= 0.75 else 4)))
        )
    )
    dataset["降幅时间点差值"] = (
        (dataset["降幅时间点"] - dataset["透析开始时间"]).dt.total_seconds() / 60 / 60
    )
    dataset["降幅时间点差值区间"] = dataset["降幅时间点差值"].apply(
        lambda x: (
            0 if x <= 0 else (1 if x <= 1 else (2 if x <= 2 else (3 if x <= 3 else 4)))
        )
    )
    dataset["透前体重-干体重"] = dataset["透前体重"] - dataset["干体重"]

    # 新增临床特征
    dataset["脉压差"] = dataset["透前收缩压"] - dataset["透前舒张压"]
    dataset["平均动脉压"] = 1/3 * dataset["透前收缩压"] + 2/3 * dataset["透前舒张压"]
    dataset["超负荷"] = (dataset["透前体重"] - dataset["干体重"]) / dataset["干体重"]
    # 超滤比: mL/h (超滤量MAX/透析时长)，临床关注 >1000 mL/h 为高风险
    dataset["超滤率_绝对"] = dataset["超滤量MAX"] / dataset["实际透析时长"]
    dataset["超滤率_体重归一化"] = dataset["超滤量MAX"] / dataset["实际透析时长"] / dataset["透前体重"]
    dataset["超滤率高危Flag"] = (dataset["超滤率_体重归一化"] > 13).astype(int)
    dataset["透析龄占比"] = dataset["透析龄"] / dataset["透析年龄"]

    # 处理异常值：合理范围截断
    dataset["超负荷"] = dataset["超负荷"].clip(-0.1, 0.3)
    dataset["透析龄占比"] = dataset["透析龄占比"].clip(0, 1)
    dataset["超滤率_绝对"] = dataset["超滤率_绝对"].clip(0, 3000)
    dataset["超滤率_体重归一化"] = dataset["超滤率_体重归一化"].clip(0, 40)
    dataset["脉压差"] = dataset["脉压差"].clip(20, 120)

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
        "涨幅时间点差值区间",
        "降幅时间点",
        "降幅时间点比值",
        "降幅时间点比值区间",
        "透前体重-干体重",
        "minutes_from_start_list",
        "duration_minutes",
        "et_min",
        "events",
        "stage_30_60_120",
        "stage_30_90_120",
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
        "透中出汗",
        "透中呼吸困难",
        "透中头晕",
        "透中头痛",
        "透中心悸",
        "透中恶心",
        "透中肌肉痉挛",
        "透中胸闷",
        "脉压差",
        "平均动脉压",
        "超负荷",
        "超滤率_绝对",
        "超滤率_体重归一化",
        "超滤率高危Flag",
        "透析龄占比",
    ]

    # Save the resulting DataFrame
    dataset = dataset[keep_columns]
    dataset.to_csv("/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/data/深医_optimized_data" + str(first_pressure_sd) + ".csv")
    # dataset.to_csv("Final_data/深医_optimized_data"+str(first_pressure_sd)".csv")
    # del X['涨幅时间点区间']
    # del X['透中高血压_计算']

    ## 继续执行历史平均值计算
    return process_historical_averages(dataset, first_pressure_sd=first_pressure_sd, use_parallel=use_parallel, n_threads=n_threads)

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


def process_historical_averages_shenyi_parallel(dataset, mean_columns, first_pressure_sd="透前动脉压", n_threads=20):
    """使用多线程处理深医历史平均值计算的函数"""
   
    
    """使用多线程处理福鼎历史平均值计算的函数"""
    dataset = dataset.sort_values(by=["患者id", "透析日期"])
    
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
    result_df.to_csv(f"/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/data/深医_result_df{first_pressure_sd}.csv")
    return result_df


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


def process_historical_rates_shenyi_parallel(dataset, n_threads=20):
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
            'history_HBP_rate',
            'history_LBP_times_0_rate',
            'history_LBP_times_1_rate',
            'history_LBP_times_2_rate',
            'history_LBP_times_3_rate',
            'history_LBP_times_4_rate',
        ],
    )
    
    return Y_rate


def process_historical_averages(dataset, first_pressure_sd='', use_parallel=True, n_threads=20):
    """处理历史平均值计算的函数（支持并行和串行模式）"""
    # 定义需要计算历史平均值的列
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
        "降幅时间点比值区间",
        "降幅时间点差值区间",
        "降幅时间点比值",
        "降幅时间点差值",
        "涨幅时间点比值",
        "涨幅时间点差值",
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
        "脉压差",
        "平均动脉压",
        "超负荷",
        "超滤率_绝对",
        "超滤率_体重归一化",
        "超滤率高危Flag",
        "透析龄占比",
    ]
    
    target_file = f'/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/data/深医_result_df{first_pressure_sd}.csv'
    
    if os.path.exists(target_file):
        print(f"检测到已存在文件: {target_file}")
        print("读取已存在的文件，跳过中间计算，直接执行下一步")
        result_df = pd.read_csv(target_file)
    else:
        if use_parallel:
            # 使用并行处理
            result_df = process_historical_averages_shenyi_parallel(dataset,mean_columns,  first_pressure_sd, n_threads)
        else:
            # 使用原有的串行处理
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
            result_df.to_csv(
                path_or_buf=("/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/data/深医_result_df" + str(first_pressure_sd) + ".csv")
            )
    
    whole_data = pd.merge(
        dataset.reset_index(drop=True), result_df.reset_index(drop=True), on=["患者id", "透析日期"], how="inner"
    )

    whole_data = whole_data.sort_values(by=["患者id", "透析日期"]).reset_index(drop=True)
    for col in mean_columns:
        values = whole_data[col].values
        patient_ids = whole_data["患者id"].values
        result = np.zeros(len(whole_data))
        cumsum = 0.0
        count = 0
        prev_pid = None
        for i in range(len(whole_data)):
            pid = patient_ids[i]
            if pid != prev_pid:
                cumsum = 0.0
                count = 0
                prev_pid = pid
            v = values[i]
            try:
                v_float = float(v)
                is_nan = np.isnan(v_float)
            except (ValueError, TypeError):
                is_nan = True
            if is_nan:
                result[i] = cumsum / count if count > 0 else 0.0
            else:
                cumsum += v_float
                count += 1
                result[i] = cumsum / count
        whole_data['历史平均' + col] = result

    # 计算历史比率
    if use_parallel:
        # 使用并行处理
        Y_rate = process_historical_rates_shenyi_parallel(dataset, n_threads)
    else:
        # 使用原有的串行处理
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

                y_rate = [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
                continue
                # y_rate = [
                #     0,
                #     0,
                #     0,
                #     0,
                #     0,
                #     0,
                # ]
            Y_rate.append(y_rate)

        Y_rate = pd.DataFrame(
            Y_rate,
            columns=[
                "history_HBP_rate",
                "history_LBP_times_0_rate",
                "history_LBP_times_1_rate",
                "history_LBP_times_2_rate",
                "history_LBP_times_3_rate",
                "history_LBP_times_4_rate",
            ],
        )

    final_data = pd.concat([whole_data, Y_rate], axis=1)

    Y_rate = []
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
        if i % 1000 == 0:
            print("加载数据", f"{str(round(i / len(dataset) * 100, 10))}%")

        p = dataset["患者id"].iloc[i]

        if last_p == p:
            history_HBP_rate = history_HBP / total_times
            history_HBP_time_rate_0 = history_LBP_times_0 / total_times
            history_HBP_time_rate_1 = history_LBP_times_1 / total_times
            history_HBP_time_rate_2 = history_LBP_times_2 / total_times
            history_HBP_time_rate_3 = history_LBP_times_3 / total_times
            history_HBP_time_rate_4 = history_LBP_times_4 / total_times

            y_rate = [
                history_HBP_rate,
                history_HBP_time_rate_0,
                history_HBP_time_rate_1,
                history_HBP_time_rate_2,
                history_HBP_time_rate_3,
                history_HBP_time_rate_4,
            ]

            if dataset["透中低血压_计算"].iloc[i] == 0:
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 4:
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

            if dataset["透中低血压_计算"].iloc[i] == 0:
                history_LBP_times_0 += 1
            if dataset["透中低血压_计算"].iloc[i] == 1:
                history_HBP += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 1:
                history_LBP_times_1 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 2:
                history_LBP_times_2 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 3:
                history_LBP_times_3 += 1
            if dataset["降幅时间点差值区间"].iloc[i] == 4:
                history_LBP_times_4 += 1

            y_rate = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]
            continue
            y_rate = [
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        Y_rate.append(y_rate)

    Y_rate = pd.DataFrame(
        Y_rate,
        columns=[
            "history_HBP",
            "history_LBP_times_0",
            "history_LBP_times_1",
            "history_LBP_times_2",
            "history_LBP_times_3",
            "history_LBP_times_4",
        ],
    )

    final_data = pd.concat([final_data, Y_rate], axis=1)

    # 清理：删除自动生成的索引列和重复列
    drop_cols = [c for c in final_data.columns if c.startswith('Unnamed')]
    if '降幅时间点比值区间.1' in final_data.columns:
        drop_cols.append('降幅时间点比值区间.1')
    if drop_cols:
        final_data.drop(columns=drop_cols, inplace=True, errors='ignore')

    final_data.to_csv(
        path_or_buf=("/home/cht/Works/domain-adaptive-deep-survival-network/data_preprocessing/data/深医_final_data.csv"),
        index=False
    )
    
    return final_data


# 主执行部分 - 使用20个线程进行并行计算
shengyi_dataset(use_parallel=True, n_threads=30)
# print(f"=== {i} 数据处理完成 ===")
