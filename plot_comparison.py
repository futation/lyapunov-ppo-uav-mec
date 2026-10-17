"""plot_comparison.py — 扫描实验对比柱状图 (合并 UAV 数量 / 到达率两种维度)

从 sweep.py 生成的汇总 config (uav_comparison_config.json 或
arrival_comparison_config.json) 加载各配置的 training_history,
以每种算法收敛后 (最后 window 个 episode 均值) 的指标绘制分组柱状图。

用法:
    python plot_comparison.py results_multi_uav/uav_comparison_config.json
    python plot_comparison.py results_multi_arrival/arrival_comparison_config.json --group-by arrival
    python plot_comparison.py <config> --window 50 --output ./figures
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import argparse
from typing import Dict, List


# ==================== 全局样式 ====================
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLOR_PPO    = '#E74C3C'     # PPO — 红
COLOR_DQN    = '#2980B9'     # DQN — 蓝
COLOR_RANDOM = '#95A5A6'     # Random — 灰
COLOR_GREEDY = '#F39C12'     # Greedy — 橙

BAR_WIDTH = 0.2
ALGO_ORDER = ['PPO', 'DQN', 'Greedy', 'Random']
COLOR_MAP = {'PPO': COLOR_PPO, 'DQN': COLOR_DQN,
             'Random': COLOR_RANDOM, 'Greedy': COLOR_GREEDY}

# 两种扫描维度的展示配置
GROUP_CONFIGS = {
    'uav': dict(config_key='uav_counts', base_name='UAV_comparison',
                xlabel='Number of UAVs'),
    'arrival': dict(config_key='arrival_rates', base_name='Arrival_comparison',
                    xlabel='Task Arrival Rate (GFLOPS/s)'),
}


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def resolve_path(json_path: str, config_dir: str) -> str:
    """解析历史 JSON 路径: 先按原样, 再相对 config 所在目录"""
    if os.path.isabs(json_path) and os.path.exists(json_path):
        return json_path
    candidate = os.path.normpath(os.path.join(config_dir, json_path))
    if os.path.exists(candidate):
        return candidate
    return json_path  # 保持原样, 让 load_json 报错


def extract_converged_metrics(history: Dict, window: int = 100) -> Dict[str, Dict[str, float]]:
    """提取收敛后指标 (最后 window 个 episode 均值)

    Returns:
        {'PPO': {'reward','delay','energy'}, 'DQN': ..., 'Random': ..., 'Greedy': ...}
    """
    results = {}

    def _mean_tail(arr):
        w = min(window, len(arr)) if arr else 0
        return float(np.mean(arr[-w:])) if w > 0 else 0.0

    results['PPO'] = {
        'reward': _mean_tail(history.get('episode_rewards', [])),
        'delay':  _mean_tail(history.get('episode_delays', [])),
        'energy': _mean_tail(history.get('episode_energies', [])),
    }

    dqn = history.get('dqn')
    if dqn is not None:
        results['DQN'] = {
            'reward': _mean_tail(dqn.get('episode_rewards', [])),
            'delay':  _mean_tail(dqn.get('episode_delays', [])),
            'energy': _mean_tail(dqn.get('episode_energies', [])),
        }

    baseline = history.get('baseline')
    if baseline is not None:
        for name, label in [('random', 'Random'), ('greedy', 'Greedy')]:
            if name in baseline:
                results[label] = {
                    'reward': _mean_tail(baseline[name].get('rewards', [])),
                    'delay':  _mean_tail(baseline[name].get('delays', [])),
                    'energy': _mean_tail(baseline[name].get('energies', [])),
                }

    return results


def _plot_single_metric(metric_key: str, ylabel: str, x_label: str,
                        group_keys: List, all_data: Dict, algorithms: List[str],
                        save_path: str):
    """绘制单个指标的分组柱状图并保存 PNG"""
    n_groups = len(group_keys)
    n_algos = len(algorithms)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(n_groups)
    offsets = np.linspace(-(n_algos - 1) * BAR_WIDTH / 2,
                          (n_algos - 1) * BAR_WIDTH / 2, n_algos)

    for algo_idx, algo_name in enumerate(algorithms):
        values = [all_data.get(g, {}).get(algo_name, {}).get(metric_key, 0.0)
                  for g in group_keys]
        ax.bar(x + offsets[algo_idx], values, BAR_WIDTH,
               color=COLOR_MAP.get(algo_name, '#888888'),
               label=algo_name, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in group_keys], fontsize=11)
    ax.set_xlabel(x_label)
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.grid(True, linestyle=':', linewidth=0.5, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    fig.tight_layout()
    fig.savefig(save_path, format='png', dpi=300)
    print(f"[OK] {save_path}")
    plt.close(fig)


def generate_all_plots(config_path: str, group_by: str = 'uav',
                       output_dir: str = "./figures", window: int = 100):
    """从汇总 config 生成 reward/delay/energy 三张对比柱状图"""
    cfg = GROUP_CONFIGS[group_by]
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] 加载配置: {config_path}")
    summary = load_json(config_path)
    if cfg['config_key'] not in summary:
        raise ValueError(f"配置文件缺少 '{cfg['config_key']}' 字段, "
                         f"是否选对了 --group-by?")
    group_map = summary[cfg['config_key']]

    config_dir = os.path.dirname(os.path.abspath(config_path))

    # ---- 加载全部训练历史并提取收敛指标 ----
    all_data = {}        # {group_value: {algo: {metric: val}}}
    algorithms_set = set()

    for key_str, json_path in group_map.items():
        # 到达率 key 是浮点数字符串, 排序用
        key_val = float(key_str) if '.' in key_str else int(key_str)
        abs_json_path = resolve_path(json_path, config_dir)
        print(f"[INFO] 加载 {key_str}: {abs_json_path}")
        try:
            history = load_json(abs_json_path)
            metrics = extract_converged_metrics(history, window)
            all_data[key_val] = metrics
            algorithms_set.update(metrics.keys())
        except Exception as e:
            print(f"[WARN] 加载 {key_str} 失败: {e}")

    if not all_data:
        print("[ERROR] 没有成功加载任何数据!")
        return

    algorithms = [a for a in ALGO_ORDER if a in algorithms_set]
    group_sorted = sorted(all_data.keys())
    print(f"[INFO] 检测到算法: {algorithms} | 分组: {group_sorted}")

    # ---- 绘制 3 张图 ----
    metrics = [('reward', 'Average Reward'), ('delay', 'Average Delay (s)'),
               ('energy', 'Average Energy (J)')]
    for metric_key, ylabel in metrics:
        save_path = os.path.join(output_dir, f"{cfg['base_name']}_{metric_key}.png")
        _plot_single_metric(metric_key, ylabel, cfg['xlabel'],
                            group_sorted, all_data, algorithms, save_path)

    # ---- 打印数值摘要 ----
    print("\n" + "=" * 70)
    print(f"  收敛指标摘要 (最后 {window} episodes 均值)")
    print("=" * 70)
    for metric_key, ylabel in metrics:
        print(f"\n  [{ylabel}]")
        header = f"  {'Group':>8}"
        for a in algorithms:
            header += f"  {a:>10}"
        print(header)
        print("  " + "-" * (10 + 12 * len(algorithms)))
        for g in group_sorted:
            row = f"  {g:>8}"
            for a in algorithms:
                val = all_data.get(g, {}).get(a, {}).get(metric_key, float('nan'))
                row += f"  {val:>10.3f}"
            print(row)

    print(f"\n[DONE] 所有图片已保存至: {output_dir}")


# ==================== CLI 入口 ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="扫描实验多算法对比柱状图 (UAV 数量 / 任务到达率)")

    parser.add_argument('config_file', type=str,
                        help='sweep.py 生成的汇总 config JSON 路径')
    parser.add_argument('--group-by', type=str, default='uav',
                        choices=['uav', 'arrival'],
                        help='扫描维度 (默认: uav)')
    parser.add_argument('--output', type=str, default='./figures',
                        help='图片输出目录 (默认: ./figures)')
    parser.add_argument('--window', type=int, default=100,
                        help='收敛窗口大小: 取最后 N 个 episode 均值 (默认: 100)')

    args = parser.parse_args()

    generate_all_plots(config_path=args.config_file, group_by=args.group_by,
                       output_dir=args.output, window=args.window)
