#!/bin/sh
# 兼容Ubuntu默认Dash Shell的SLAM批量执行脚本
# 移除bash数组，改用字符串拆分+case分支，避免语法错误

# 设置CUDA设备（固定为1）
export CUDA_VISIBLE_DEVICES=1

# 创建日志目录（如果不存在）
LOG_DIR="./slam_run_logs"
mkdir -p ${LOG_DIR}

# 遍历所有实验名称（按顺序排列，Dash完全支持）
for exp_name in room0 room1 room2 office0 office1 office2 office3 office4 fr1 fr2 fr3; do
    # 根据实验名称匹配对应的配置文件路径
    case ${exp_name} in
        room0) config_path="./configs/rgbd/replica/room0.yaml" ;;
        room1) config_path="./configs/rgbd/replica/room1.yaml" ;;
        room2) config_path="./configs/rgbd/replica/room2.yaml" ;;
        office0) config_path="./configs/rgbd/replica/office0.yaml" ;;
        office1) config_path="./configs/rgbd/replica/office1.yaml" ;;
        office2) config_path="./configs/rgbd/replica/office2.yaml" ;;
        office3) config_path="./configs/rgbd/replica/office3.yaml" ;;
        office4) config_path="./configs/rgbd/replica/office4.yaml" ;;
        fr1) config_path="./configs/rgbd/tum/fr1_desk.yaml" ;;
        fr2) config_path="./configs/rgbd/tum/fr2_xyz.yaml" ;;
        fr3) config_path="./configs/rgbd/tum/fr3_office.yaml" ;;
    esac
    
    # 定义日志文件（带时间戳）
    log_file="${LOG_DIR}/${exp_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # 打印开始信息
    echo "========================================"
    echo "开始执行实验: ${exp_name}"
    echo "配置文件: ${config_path}"
    echo "日志文件: ${log_file}"
    echo "========================================"
    
    # 执行SLAM命令，并重定向输出到日志
    python slam.py --config ${config_path} --eval > ${log_file} 2>&1
    
    # 检查命令执行状态
    if [ $? -eq 0 ]; then
        echo "✅ 实验 ${exp_name} 执行成功！"
    else
        echo "❌ 实验 ${exp_name} 执行失败！请查看日志: ${log_file}"
        # 可选：如果希望某个实验失败后停止整体执行，取消下面的注释
        # exit 1
    fi
    
    echo ""  # 空行分隔不同实验的输出
done

# 所有实验执行完成
echo "🎉 所有实验执行完毕！日志文件保存在: ${LOG_DIR}"

