#!/bin/bash

# 脚本路径
SCRIPT_PATH="/home/cht/Works/PredictionTimeHypotensionDialysis/透前模型/透前数据__透中低血压建模.py"
LOG_FILE="/home/cht/Works/PredictionTimeHypotensionDialysis/model_run.log"
ERROR_LOG="/home/cht/Works/PredictionTimeHypotensionDialysis/model_error.log"

# 记录开始时间
echo "[$(date)] 开始运行模型脚本" >> "$LOG_FILE"

# 无限循环，直到成功运行
while true; do
    echo "[$(date)] 尝试运行脚本: $SCRIPT_PATH" >> "$LOG_FILE"
    
    # 运行Python脚本
    cd "/home/cht/Works/PredictionTimeHypotensionDialysis"
    python3 "$SCRIPT_PATH" >> "$LOG_FILE" 2>> "$ERROR_LOG"
    
    # 检查退出状态
    if [ $? -eq 0 ]; then
        echo "[$(date)] 脚本成功完成" >> "$LOG_FILE"
        break
    else
        echo "[$(date)] 脚本运行失败，30秒后重试" >> "$LOG_FILE"
        echo "[$(date)] 错误详情请查看: $ERROR_LOG" >> "$LOG_FILE"
        sleep 30
    fi
done

echo "[$(date)] 任务完成" >> "$LOG_FILE"