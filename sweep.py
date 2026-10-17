"""sweep.py — 参数扫描自动化训练脚本

依次在多组配置下运行完整实验 (PPO 训练 + DQN 基线 + Random/Greedy 基准),
每组配置保存独立的 training_history JSON, 并生成汇总 config 供
plot_comparison.py 绘制对比柱状图。

两种扫描模式:
    python sweep.py --sweep uav          # UAV 数量: 10/20/30/40/50 (5:3:2 缩放)
    python sweep.py --sweep arrival      # 任务到达率: 45/47.5/50/52.5 GFLOPS/s (UAV=10)
    python sweep.py --sweep uav --quick  # 快速验证模式 (50 episodes, 跳过 DQN)
"""

import numpy as np
import torch
import os
import sys
import json
import time
import argparse
from datetime import datetime

from config import NUM_EPISODES, STEPS_PER_EPISODE, UPDATE_INTERVAL, LAMBDA_ARRIVAL
from train import run_training, set_seed


# ==================== 扫描配置 ====================
# UAV 数量 -> (低, 中, 高) 算力档配比 (5:3:2 等比缩放)
UAV_CONFIGS = {
    10:  (5,  3,  2),
    20:  (10, 6,  4),
    30:  (15, 9,  6),
    40:  (20, 12, 8),
    50:  (25, 15, 10),
}

# 到达率扫描 (UAV 固定 10)
ARRIVAL_RATES = [45.0, 47.5, 50.0, 52.5]


def _run_one_config(label: str, config_save_dir: str, run_kwargs: dict) -> str:
    """运行单组配置并返回生成的 training_history JSON 路径 (未找到返回 None)"""
    os.makedirs(config_save_dir, exist_ok=True)
    history = run_training(save_dir=config_save_dir, **run_kwargs)

    json_files = [f for f in os.listdir(config_save_dir)
                  if f.startswith('training_history_') and f.endswith('.json')]
    if json_files:
        json_files.sort(reverse=True)  # 按文件名时间戳取最新
        return os.path.join(config_save_dir, json_files[0])
    print(f"[WARN] {label} 训练完成但未找到 JSON 文件")
    return None


def sweep_uav(uav_list, num_episodes, steps_per_episode, update_interval,
              base_save_dir, device, run_dqn, seed):
    """扫描不同 UAV 数量 (5:3:2 算力配比等比缩放)"""
    os.makedirs(base_save_dir, exist_ok=True)
    json_paths = {}
    total_start = time.time()

    for i, num_uavs in enumerate(uav_list):
        low, mid, high = UAV_CONFIGS[num_uavs]
        label = f"UAV{num_uavs}"
        print("\n" + "=" * 70)
        print(f"  [{i+1}/{len(uav_list)}] 训练配置: {label} "
              f"(低{low} + 中{mid} + 高{high})")
        print("=" * 70)

        set_seed(seed)
        try:
            path = _run_one_config(label, os.path.join(base_save_dir, f"uav{num_uavs}"),
                                   dict(num_episodes=num_episodes,
                                        steps_per_episode=steps_per_episode,
                                        update_interval=update_interval,
                                        device=device, run_baselines=True, run_dqn=run_dqn,
                                        low_uavs=low, mid_uavs=mid, max_uavs=high))
            if path:
                json_paths[str(num_uavs)] = path
                print(f"[OK] {label} 训练完成")
        except Exception as e:
            print(f"[ERROR] {label} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - total_start
        remaining = elapsed / (i + 1) * (len(uav_list) - i - 1)
        print(f"[INFO] 已用时: {elapsed/60:.1f} min, 预计剩余: {remaining/60:.1f} min")

    _finish(base_save_dir, "uav_comparison_config.json",
            "不同 UAV 数量下的训练结果 (PPO + DQN + Random + Greedy)",
            "uav_counts", json_paths, total_start)
    return json_paths


def sweep_arrival(arrival_rates, num_episodes, steps_per_episode, update_interval,
                  base_save_dir, device, run_dqn, seed):
    """扫描不同任务到达率 (UAV 固定 10 架)"""
    os.makedirs(base_save_dir, exist_ok=True)
    json_paths = {}
    total_start = time.time()

    for i, rate in enumerate(arrival_rates):
        label = f"rate_{rate}"
        print("\n" + "=" * 70)
        print(f"  [{i+1}/{len(arrival_rates)}] 训练配置: {label} GFLOPS/s (UAV=10)")
        print("=" * 70)

        set_seed(seed)
        try:
            path = _run_one_config(label, os.path.join(base_save_dir, f"rate_{rate}"),
                                   dict(num_episodes=num_episodes,
                                        steps_per_episode=steps_per_episode,
                                        update_interval=update_interval,
                                        device=device, run_baselines=True, run_dqn=run_dqn,
                                        task_arrive_rate=rate))
            if path:
                json_paths[str(rate)] = path
                print(f"[OK] {label} 训练完成")
        except Exception as e:
            print(f"[ERROR] {label} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - total_start
        remaining = elapsed / (i + 1) * (len(arrival_rates) - i - 1)
        print(f"[INFO] 已用时: {elapsed/60:.1f} min, 预计剩余: {remaining/60:.1f} min")

    _finish(base_save_dir, "arrival_comparison_config.json",
            "不同任务到达率下的训练结果 (PPO + DQN + Random + Greedy)",
            "arrival_rates", json_paths, total_start)
    return json_paths


def _finish(base_save_dir, config_name, description, key, json_paths, total_start):
    """生成汇总配置文件并打印总结"""
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 70)
    print(f"  全部训练完成! 总耗时: {total_elapsed/60:.1f} min")
    print("=" * 70)

    config_path = os.path.join(base_save_dir, config_name)
    config_data = {
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        key: json_paths,
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] 配置文件已保存: {config_path}")


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="参数扫描自动化训练 (PPO + DQN + Random/Greedy 基准)")

    parser.add_argument('--sweep', type=str, default='uav', choices=['uav', 'arrival'],
                        help='扫描维度: uav=UAV 数量, arrival=任务到达率 (默认: uav)')
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                        help=f'每个配置的训练 episode 数 (默认: {NUM_EPISODES})')
    parser.add_argument('--steps', type=int, default=STEPS_PER_EPISODE,
                        help=f'每 episode 步数 (默认: {STEPS_PER_EPISODE})')
    parser.add_argument('--update', type=int, default=UPDATE_INTERVAL,
                        help=f'PPO 更新间隔 (默认: {UPDATE_INTERVAL})')
    parser.add_argument('--uavs', type=str, default='10,20,30,40,50',
                        help='(uav 模式) 要训练的 UAV 数量, 逗号分隔')
    parser.add_argument('--rates', type=str, default=','.join(str(r) for r in ARRIVAL_RATES),
                        help='(arrival 模式) 到达率列表, 逗号分隔')
    parser.add_argument('--save', type=str, default=None,
                        help='结果保存根目录 (默认: ./results_multi_<mode>)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--no-dqn', action='store_true', default=False,
                        help='跳过 DQN 基线训练')
    parser.add_argument('--quick', action='store_true', default=False,
                        help='快速验证模式: 最多 50 episodes, 跳过 DQN')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')

    args = parser.parse_args()

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    episodes = args.episodes
    run_dqn = not args.no_dqn
    if args.quick:
        episodes = min(episodes, 50)
        run_dqn = False
        print("[QUICK MODE] 最多 50 episodes, 跳过 DQN")

    print("=" * 70)
    print(f"  参数扫描自动化训练 ({args.sweep})")
    print("=" * 70)
    print(f"  Episodes/配置: {episodes} | DQN: {run_dqn} | 设备: {device}")

    if args.sweep == 'uav':
        uav_list = [int(x.strip()) for x in args.uavs.split(',')]
        for n in uav_list:
            if n not in UAV_CONFIGS:
                print(f"[ERROR] 不支持的 UAV 数量: {n}. 支持: {list(UAV_CONFIGS.keys())}")
                sys.exit(1)
        print(f"  UAV 配置: {uav_list}")
        save_dir = args.save or './results_multi_uav'
        sweep_uav(uav_list, episodes, args.steps, args.update,
                  save_dir, device, run_dqn, args.seed)
    else:
        rates = [float(x.strip()) for x in args.rates.split(',')]
        print(f"  到达率: {rates}")
        save_dir = args.save or './results_multi_arrival'
        sweep_arrival(rates, episodes, args.steps, args.update,
                      save_dir, device, run_dqn, args.seed)
