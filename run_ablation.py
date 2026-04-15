import yaml
import subprocess
import os
import time
from copy import deepcopy

# =========================================================================
# 配置区域
# =========================================================================
# 要执行的 YAML 配置文件路径 (你可以根据需要改成 fr3_office.yaml)
#CONFIG_FILE_PATH = "./configs/rgbd/replica/room0.yaml"
# 日志保存目录
LOG_DIR = "./ablation_logs"
# GPU 设置 (默认使用 GPU 1)
CUDA_DEVICE = "1"

# 定义所有消融实验的状态字典
# 每个字典定义了在 ablation 中应该为 True 的字段，其余默认设为 False
ABLATION_EXPERIMENTS = {
    "ExpB_2DGS_Only": {
        "use_color_refinement": True,
        "use_submap": False,
        "use_loop_closure": False,
        "use_fdn": False,
        "use_fft_mask": False,
        "use_error_mask": False
    },
    "ExpC_Plus_Submap": {
        "use_color_refinement": True,
        "use_submap": True,
        "use_loop_closure": False,
        "use_fdn": False,
        "use_fft_mask": False,
        "use_error_mask": False
    },
    "ExpD_Plus_LoopClosure": {
        "use_color_refinement": True,
        "use_submap": True,
        "use_loop_closure": True,
        "use_fdn": False,
        "use_fft_mask": False,
        "use_error_mask": False
    },
    "ExpE_Plus_FDN": {
        "use_color_refinement": True,
        "use_submap": True,
        "use_loop_closure": True,
        "use_fdn": True,
        "use_fft_mask": False,
        "use_error_mask": False
    },
    "ExpF_Plus_FFTMask": {
        "use_color_refinement": True,
        "use_submap": True,
        "use_loop_closure": True,
        "use_fdn": True,
        "use_fft_mask": True,
        "use_error_mask": False
    },
    "ExpG_Full_System": {
        "use_color_refinement": True,
        "use_submap": True,
        "use_loop_closure": True,
        "use_fdn": True,
        "use_fft_mask": True,
        "use_error_mask": True
    }
}


def update_yaml_config(file_path, ablation_settings):
    """读取 YAML 文件，更新 Ablation 字段，并安全写回"""
    with open(file_path, 'r', encoding='utf-8') as f:
        # 使用 safe_load 防止执行恶意代码
        config = yaml.safe_load(f)

    # 确保 Ablation 字段存在
    if 'Ablation' not in config:
        config['Ablation'] = {}

    # 更新消融配置
    for key, value in ablation_settings.items():
        config['Ablation'][key] = value

    # 写回 YAML (保留原有的列表/字典结构格式)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"[*] YAML 配置文件已更新: {file_path}")


def run_slam(exp_name):
    """执行 SLAM 脚本并将终端输出保存到文件"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, f"{exp_name}.log")

    print(f"\n{'=' * 60}")
    print(f"🚀 开始执行实验: {exp_name}")
    print(f"📄 日志将保存至: {log_file_path}")
    print(f"{'=' * 60}\n")

    # 构建执行命令
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE

    # 使用 stdbuf 禁用缓冲，确保进度条和日志能实时写入文件
    command = ["python", "slam.py", "--config", CONFIG_FILE_PATH, "--eval"]

    with open(log_file_path, "w", encoding='utf-8') as log_file:
        # Popen 允许我们同时在屏幕上看到输出，并写入文件
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将标准错误合并到标准输出
            text=True,
            bufsize=1
        )

        # 实时读取并打印，同时写入文件
        for line in process.stdout:
            print(line, end="")  # 终端打印
            log_file.write(line)  # 写入日志文件
            log_file.flush()  # 强制刷入硬盘，防止崩溃时日志丢失

        process.wait()

    if process.returncode == 0:
        print(f"\n✅ 实验 {exp_name} 执行完毕！\n")
    else:
        print(f"\n❌ 实验 {exp_name} 执行失败，退出码: {process.returncode}。请检查日志。\n")

    # 给系统喘息的时间，清理显存
    print("⏳ 等待 5 秒释放显存...")
    time.sleep(5)


if __name__ == "__main__":
    print("🚀 自动化消融实验脚本启动...")
    print(f"📌 目标配置文件: {CONFIG_FILE_PATH}")

    # 遍历所有定义的实验并依次执行
    for exp_name, settings in ABLATION_EXPERIMENTS.items():
        print(f"\n>>>>> 准备执行 {exp_name} <<<<<")
        print(f"当前配置策略: {settings}")

        # 1. 更新配置文件
        update_yaml_config(CONFIG_FILE_PATH, settings)

        # 2. 运行 SLAM
        run_slam(exp_name)

    print("\n🎉 所有消融实验已全部完成！辛苦了！")
    print(f"📂 请前往 {LOG_DIR} 目录查看收集到的 ATE、PSNR 和 Peak Memory 数据。")