import numpy as np
import ast


def convert_to_float_list(input_list):
    if input_list is None:
        return None
    iterable = None
    if isinstance(input_list, list):
        iterable = input_list
    elif isinstance(input_list, (int, float)):
        iterable = [input_list]
    elif isinstance(input_list, str):
        s = input_list.strip()
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            parsed = s
        if isinstance(parsed, list):
            iterable = parsed
        elif isinstance(parsed, (int, float)):
            iterable = [parsed]
        else:
            try:
                iterable = [float(s)]
            except Exception:
                return None
    else:
        return None
    out = []
    for value in iterable:
        if value is None:
            continue
        if isinstance(value, str):
            v = value.strip()
            if v == "" or v.upper() == "NA":
                continue
            try:
                out.append(float(v))
            except Exception:
                continue
        else:
            try:
                out.append(float(value))
            except Exception:
                continue
    return out if out else None


def replace_extremes_with_percentiles(
    dataset, list_columns, lower_percentile=5, upper_percentile=95
):
    # 按列单独裁剪，避免将不同量纲的动态变量混在一起计算统一阈值。
    for col in list_columns:
        column_values = [value for lst in dataset[col] if lst for value in lst]
        if not column_values:
            continue

        lower_percentile_val = np.percentile(column_values, lower_percentile)
        upper_percentile_val = np.percentile(column_values, upper_percentile)

        def replace_extremes(lst):
            if lst:
                return [
                    (
                        lower_percentile_val
                        if value < lower_percentile_val
                        else (
                            upper_percentile_val
                            if value > upper_percentile_val
                            else value
                        )
                    )
                    for value in lst
                ]
            return lst

        dataset[col] = dataset[col].apply(replace_extremes)

    return dataset


# def convert_to_float_list(pressure_str):
#     # 检查是否为 NaN
#     if pd.isna(pressure_str):
#         return []

#     # 检查是否为字符串类型
#     if isinstance(pressure_str, str):
#         # 预处理数据，去除方括号和空格，并用逗号分割
#         pressure_str = pressure_str.replace("[", "").replace("]", "").replace(" ", "")
#         pressure_values = pressure_str.split(",")
#         # 尝试将字符串转换为浮点数列表
#         try:
#             return [float(p) for p in pressure_values]
#         except ValueError:
#             # 如果转换失败，则返回空列表或者做其他适当的处理
#             return pressure_str
#     # 如果不是字符串类型，则返回空列表或者原始值
#     return []


# Define a function to diagnose hypertension
def diagnose_hypertension(diagnosis):
    return 1 if isinstance(diagnosis, str) and "高血压" in diagnosis else 0


def calculate_pressure_change(row, pressure_type="hypertension", first_pressure=None):
    """
    Calculate whether there is a significant pressure change.

    Parameters:
    - row: Dictionary containing the row data.
    - pressure_type: 'hypertension' or 'hypotension' to specify the type of pressure change to check.
    - first_pressure_column: Optional column name to use for the initial pressure value.

    Returns:
    - 1 if the condition for the specified pressure type is met, otherwise 0.
    """
    if pressure_type == "hypertension":
        pressures = convert_to_float_list(row["动脉压"])
        threshold = 10
        comparator = lambda p, fp: p - fp > threshold
    elif pressure_type == "hypotension":
        pressures = convert_to_float_list(row["透析中收缩压"])
        threshold = 30
        comparator = lambda p, fp: (fp - p >= threshold) or p <= 90
    # elif pressure_type == "hypotension":
    #     pressures = convert_to_float_list(row["动脉压"])
    #     threshold = 20
    #     comparator = lambda p, fp: (fp - p >= threshold)

    # first_pressure = pressures[0]
    if first_pressure is None:
        first_pressure = pressures[0]
    else:
        first_pressure = row[first_pressure]

    for pressure in pressures:
        if comparator(pressure, first_pressure):
            return 1
    return 0


def calculate_time_points(row, pressure_type="hypertension", first_pressure=None):
    """
    Calculate the time point when a significant pressure change occurs.

    Parameters:
    - row: Dictionary containing the row data.
    - pressure_type: 'hypertension' or 'hypotension' to specify the type of pressure change to check.
    - first_pressure_column: Optional column name to use for the initial pressure value.

    Returns:
    - The time point when the specified condition is met, otherwise None.
    """
    # 根据pressure_type检查相应的计算结果
    if pressure_type == "hypertension" and row.get("透中高血压_计算", 0) == 0:
        return row["透中数据记录时间节点"][0]
    elif pressure_type == "hypotension" and row.get("透中低血压_计算", 0) == 0:
        return row["透中数据记录时间节点"][0]

    if pressure_type == "hypertension":
        pressures = convert_to_float_list(row["动脉压"])
        threshold = 10
        comparator = lambda p, fp: p - fp > threshold
    elif pressure_type == "hypotension":
        pressures = convert_to_float_list(row["透析中收缩压"])
        threshold = 30
        comparator = lambda p, fp: (fp - p >= threshold) or p <= 90
    # elif pressure_type == "hypotension":
    #     pressures = convert_to_float_list(row["动脉压"])
    #     threshold = 10
    #     comparator = lambda p, fp: (fp - p >= threshold)

    # first_pressure = pressures[0]
    if first_pressure is None:
        first_pressure = pressures[0]
    else:
        first_pressure = row[first_pressure]

    for i, pressure in enumerate(pressures):
        if comparator(pressure, first_pressure):
            return row["透中数据记录时间节点"][i]
    return row["透中数据记录时间节点"][0]


# Usage example:
# row = {
#     "动脉压": "120,130,140,135",
#     "透析中收缩压": "100,95,90,85",
#     "透前收缩压": 110,
#     "透中数据记录时间节点": ["08:00", "08:30", "09:00", "09:30"],
#     "透中高血压_计算": 1  # example, this will be computed in actual use
# }

# Calculate pressure change
# pressure_change_result = calculate_pressure_change(row, pressure_type='hypertension', first_pressure_column="透前收缩压")
# pressure_change_time_point = calculate_time_points(row, pressure_type='hypertension', first_pressure_column="透前收缩压")

# Calculate hypotension
# hypotension_result = calculate_pressure_change(row, pressure_type='hypotension', first_pressure_column="透前收缩压")
# hypotension_time_point = calculate_time_points(row, pressure_type='hypotension', first_pressure_column="透前收缩压")
