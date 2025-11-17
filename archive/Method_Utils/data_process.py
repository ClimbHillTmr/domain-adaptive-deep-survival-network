import pandas as pd
import numpy as np


def convert_to_float_list(input_list):
    if isinstance(input_list, str):
        try:
            parsed_list = eval(input_list)
            if isinstance(parsed_list, list):
                lst = [
                    float(value)
                    for value in parsed_list
                    if isinstance(value, (int, float, str))
                    and value
                    and value != "NA"
                    and (
                        isinstance(value, (int, float))
                        or (isinstance(value, str) and value.isdigit())
                    )
                ]
                return [x for x in lst if x is not None]
        except (SyntaxError, ValueError):
            pass
    elif isinstance(input_list, list):
        lst = [
            float(value)
            for value in input_list
            if isinstance(value, (int, float, str))
            and value
            and value != "NA"
            and (
                isinstance(value, (int, float))
                or (isinstance(value, str) and value.isdigit())
            )
        ]
        return [x for x in lst if x is not None]

    return None


def replace_extremes_with_percentiles(
    dataset, list_columns, lower_percentile=1, upper_percentile=99
):
    # 计算整个列的1%和99%分位数
    all_values = [
        value for col in list_columns for lst in dataset[col] if lst for value in lst
    ]
    lower_percentile_val = np.percentile(all_values, lower_percentile)
    upper_percentile_val = np.percentile(all_values, upper_percentile)

    # 定义函数，替换行中的极值为整列的分位数
    def replace_extremes(lst):
        if lst:
            return [
                lower_percentile_val
                if value < lower_percentile_val
                else upper_percentile_val
                if value > upper_percentile_val
                else value
                for value in lst
            ]
        return lst

    # 对每列中的极值进行替换
    for col in list_columns:
        dataset[col] = dataset[col].apply(replace_extremes)

    return dataset


# Define a function to diagnose hypertension
def diagnose_hypertension(diagnosis):
    return 1 if isinstance(diagnosis, str) and "高血压" in diagnosis else 0


# Define a function to calculate '透中高血压_计算' column
def calculate_column(row, first_pressure=None):
    pressures = convert_to_float_list(row["动脉压"])
    if first_pressure is None:
        first_pressure = pressures[0]
    else:
        first_pressure = row[first_pressure]

    # 透前动脉压纳入first

    for pressure in pressures:
        if pressure - first_pressure > 15:
            return 1
    return 0


# Define a function to calculate '涨幅时间点' column
def calculate_time_points(row, first_pressure=None):
    if row["透中高血压_计算"] == 0:
        return 0
    pressures = convert_to_float_list(row["动脉压"])
    if first_pressure is None:
        first_pressure = pressures[0]
    else:
        first_pressure = row[first_pressure]
    # pressures = row["动脉压"]
    time_points = []

    for i, pressure in enumerate(pressures[0:], start=0):
        if pressure - first_pressure > 15:
            return row["透中数据记录时间节点"][i]
    return time_points
