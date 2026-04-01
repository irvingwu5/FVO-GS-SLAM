#!/bin/sh
# 兼容Ubuntu默认Dash Shell的SLAM批量执行脚本
# 串行执行：前一个程序完全结束后才启动下一个，杜绝OOM干扰

# 设置CUDA设备（固定为1）
export CUDA_VISIBLE_DEVICES=1

# 创建日志目录（如果不存在）
LOG_DIR="./slam_run_logs"
mkdir -p ${LOG_DIR}

# 遍历所有实验名称（按顺序排列，Dash完全支持）
for exp_name in room0 room1 room2 office1 office2 office3 office4; do
    # 根据实验名称匹配对应的配置文件路径
    case ${exp_name} in
        room0) config_path="./configs/rgbd/replica/room0.yaml" ;;
        room1) config_path="./configs/rgbd/replica/room1.yaml" ;;
        room2) config_path="./configs/rgbd/replica/room2.yaml" ;;
        office1) config_path="./configs/rgbd/replica/office1.yaml" ;;
        office2) config_path="./configs/rgbd/replica/office2.yaml" ;;
        office3) config_path="./configs/rgbd/replica/office3.yaml" ;;
        office4) config_path="./configs/rgbd/replica/office4.yaml" ;;
        # 兜底：未知实验直接跳过
        *) echo "⚠️  未知实验 ${exp_name}，跳过" && continue ;;
    esac
    
    # 定义日志文件（时间戳精确到秒，防止1秒内多次运行覆盖日志）
    log_file="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # 打印开始信息
    echo "========================================"
    echo "开始执行实验: ${exp_name}"
    echo "配置文件: ${config_path}"
    echo "日志文件: ${log_file}"
    echo "========================================"
    
    # ===================== 核心：串行执行 =====================
    # 等待python进程完全结束后，才会执行后续代码
    # 2>&1 标准错误重定向，完整保存崩溃日志
    python slam.py --config ${config_path} --eval > ${log_file} 2>&1
    
    # 检查命令执行状态
    if [ $? -eq 0 ]; then
        echo "✅ 实验 ${exp_name} 执行成功！"
    else
        echo "❌ 实验 ${exp_name} 执行失败！请查看日志: ${log_file}"
        # 强制释放显存/内存，避免残留占用影响下一个实验
        sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
        # 可选：失败立即停止全部任务（推荐开启，防止OOM连锁崩溃）
        echo "🛑 检测到失败，终止所有后续实验"
        exit 1
    fi
    
    # 每个任务结束后强制释放显存/内存，彻底杜绝OOM
    echo "🧹 清理内存/显存缓存..."
    sync && echo 3 > /proc/sys/vm/drop_caches 2>/dev/null
    
    echo -e "\n"  # 空行分隔
done

# 所有实验执行完成
echo "🎉 所有实验执行完毕！日志文件保存在: ${LOG_DIR}"

