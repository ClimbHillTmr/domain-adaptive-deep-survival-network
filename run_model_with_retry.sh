#!/bin/bash

# 定义脚本路径数组
SCRIPT_PATHS=(
    "/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型/透前数据__透中低血压建模.py"
    "/home/cht/Works/PredictionTimeHypotensionDialysis/模型2.1/完整数据2_1__透中低血压建模.py"
    "/home/cht/Works/PredictionTimeHypotensionDialysis/模型2.2/完整数据2_2__透中低血压建模.py"
)

# 定义对应的日志文件名
LOG_NAMES=(
    "透前模型"
    "模型2.1"
    "模型2.2"
)

# 主日志文件
MAIN_LOG="/home/cht/Works/PredictionTimeHypotensionDialysis/model_run_all.log"

# 记录开始时间
echo "[$(date)] 开始运行所有模型脚本" >> "$MAIN_LOG"

# 运行函数
run_script_with_retry() {
    local script_path="$1"
    local model_name="$2"
    local log_file="/home/cht/Works/PredictionTimeHypotensionDialysis/model_run_${model_name}.log"
    local error_log="/home/cht/Works/PredictionTimeHypotensionDialysis/model_error_${model_name}.log"
    
    echo "[$(date)] 开始运行 ${model_name} 模型" >> "$MAIN_LOG"
    echo "[$(date)] 日志文件: ${log_file}" >> "$MAIN_LOG"
    echo "[$(date)] 错误日志: ${error_log}" >> "$MAIN_LOG"
    
    # 无限循环，直到成功运行
    while true; do
        echo "[$(date)] 尝试运行脚本: $script_path" >> "$log_file"
        echo "[$(date)] 尝试运行 ${model_name}: $script_path" >> "$MAIN_LOG"
        
        # 运行Python脚本
        cd "/home/cht/Works/PredictionTimeHypotensionDialysis"
        python3 "$script_path" >> "$log_file" 2>> "$error_log"
        
        # 检查退出状态
        if [ $? -eq 0 ]; then
            echo "[$(date)] ${model_name} 脚本成功完成" >> "$log_file"
            echo "[$(date)] ${model_name} 脚本成功完成" >> "$MAIN_LOG"
            break
        else
            echo "[$(date)] ${model_name} 脚本运行失败，30秒后重试" >> "$log_file"
            echo "[$(date)] ${model_name} 脚本运行失败，30秒后重试" >> "$MAIN_LOG"
            echo "[$(date)] 错误详情请查看: $error_log" >> "$log_file"
            sleep 30
        fi
    done
}

# 顺序运行所有脚本
for i in "${!SCRIPT_PATHS[@]}"; do
    script_path="${SCRIPT_PATHS[$i]}"
    model_name="${LOG_NAMES[$i]}"
    
    echo "[$(date)] ========== 开始运行 ${model_name} =========="
    echo "[$(date)] ========== 开始运行 ${model_name} ==========" >> "$MAIN_LOG"
    
    run_script_with_retry "$script_path" "$model_name"
    
    echo "[$(date)] ========== ${model_name} 完成 ==========" >> "$MAIN_LOG"
    echo "[$(date)] ========== ${model_name} 完成 =========="
done

echo "[$(date)] 所有模型任务完成" >> "$MAIN_LOG"
echo "[$(date)] 所有模型任务完成"