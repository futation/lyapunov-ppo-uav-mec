"""
plot_bars.py — 算法对比柱状图

生成 3 张柱状图 (Reward / Delay / Energy)
每张图: 按窗口分组, 4 种算法并排 (PPO / DQN / Random / Greedy)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from typing import Dict, Optional
from datetime import datetime


# ==================== 样式 ====================
plt.rcParams.update({
    'font.size': 13,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
})

# 颜色方案
COLORS = {
    'ppo':    '#2980B9',   # 蓝色 — PPO
    'dqn':    '#27AE60',   # 绿色 — DQN
    'random': '#95A5A6',   # 灰色 — Random
    'greedy': '#F39C12',   # 橙色 — Greedy
}
LABELS = {
    'ppo':    'PPO',
    'dqn':    'DQN',
    'random': 'Random',
    'greedy': 'Greedy',
}


def _compute_window_means(data: list, window: int = 100) -> list:
    """将数据按窗口分组，返回每组的均值列表"""
    means = []
    num_windows = len(data) // window
    for i in range(num_windows):
        start = i * window
        end   = start + window
        means.append(float(np.mean(data[start:end])))
    return means


def plot_comparison_bar_charts(
    ppo_history: Dict,
    baseline_results: Dict,
    save_dir: str,
    dqn_history: Optional[Dict] = None,
    window: int = 100,
):
    """
    绘制算法对比柱状图 (3 子图 × 4 算法)
    
    Args:
        ppo_history:      PPO 训练历史
        baseline_results: 随机和贪心结果 {'random': {...}, 'greedy': {...}}
        save_dir:         图片保存目录
        dqn_history:      DQN 训练历史 (可选, 为 None 则跳过 DQN 柱子)
        window:           分组窗口大小 (默认 100 episode)
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ---- 计算各算法窗口均值 ----
    num_episodes = len(ppo_history['episode_rewards'])
    num_windows  = num_episodes // window
    if num_windows < 1:
        print(f"[WARN] Episode 数 ({num_episodes}) 不足一个窗口 ({window}), 无法生成柱状图")
        return
    
    window_labels = [f'{i*window+1}\n~\n{(i+1)*window}' for i in range(num_windows)]
    
    # PPO
    ppo_r = _compute_window_means(ppo_history['episode_rewards'],  window)
    ppo_d = _compute_window_means(ppo_history['episode_delays'],   window)
    ppo_e = _compute_window_means(ppo_history['episode_energies'], window)
    
    # DQN
    has_dqn = dqn_history is not None
    if has_dqn:
        dqn_r = _compute_window_means(dqn_history['episode_rewards'],  window)
        dqn_d = _compute_window_means(dqn_history['episode_delays'],   window)
        dqn_e = _compute_window_means(dqn_history['episode_energies'], window)
    
    # Random & Greedy: 整体均值 (不变化, 所有窗口使用同一值)
    rand_r = float(np.mean(baseline_results['random']['rewards']))
    rand_d = float(np.mean(baseline_results['random']['delays']))
    rand_e = float(np.mean(baseline_results['random']['energies']))
    gred_r = float(np.mean(baseline_results['greedy']['rewards']))
    gred_d = float(np.mean(baseline_results['greedy']['delays']))
    gred_e = float(np.mean(baseline_results['greedy']['energies']))
    
    # ---- 绘制 3 子图 ----
    metrics = [
        ('Reward',   ppo_r,  dqn_r if has_dqn else None,  rand_r,  gred_r,  'Average Reward',       'higher'),
        ('Delay',    ppo_d,  dqn_d if has_dqn else None,  rand_d,  gred_d,  'Average Delay (s)',     'lower'),
        ('Energy',   ppo_e,  dqn_e if has_dqn else None,  rand_e,  gred_e,  'Average Energy (J)',    'lower'),
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    for idx, (metric_name, ppo_vals, dqn_vals, rand_val, gred_val, ylabel, direction) in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(num_windows)
        width = 0.2  # 每根柱子宽度
        
        # 4 组柱子
        bars_ppo = ax.bar(x - 1.5*width, ppo_vals, width, 
                          color=COLORS['ppo'], label=LABELS['ppo'],
                          edgecolor='white', linewidth=0.5, zorder=3)
        
        if dqn_vals is not None:
            bars_dqn = ax.bar(x - 0.5*width, dqn_vals, width,
                              color=COLORS['dqn'], label=LABELS['dqn'],
                              edgecolor='white', linewidth=0.5, zorder=3)
        
        bars_rand = ax.bar(x + 0.5*width, [rand_val]*num_windows, width,
                           color=COLORS['random'], label=LABELS['random'],
                           edgecolor='white', linewidth=0.5, zorder=3)
        
        bars_gred = ax.bar(x + 1.5*width, [gred_val]*num_windows, width,
                           color=COLORS['greedy'], label=LABELS['greedy'],
                           edgecolor='white', linewidth=0.5, zorder=3)
        
        # 柱上标注数值 (PPO 加粗, DQN 常规)
        for bar in bars_ppo:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            offset = 3 if h >= 0 else -3
            ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, offset), textcoords="offset points",
                       ha='center', va=va, fontsize=7,
                       fontweight='bold', color=COLORS['ppo'])
        
        if dqn_vals is not None:
            for bar in bars_dqn:
                h = bar.get_height()
                va = 'bottom' if h >= 0 else 'top'
                offset = 3 if h >= 0 else -3
                ax.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width()/2, h),
                           xytext=(0, offset), textcoords="offset points",
                           ha='center', va=va, fontsize=7, color=COLORS['dqn'])
        
        # 标注 Random 值 (仅第一组)
        va_rand = 'bottom' if rand_val >= 0 else 'top'
        off_rand = 3 if rand_val >= 0 else -3
        ax.annotate(f'{rand_val:.1f}', 
                   xy=(bars_rand[0].get_x() + bars_rand[0].get_width()/2, rand_val),
                   xytext=(0, off_rand), textcoords="offset points",
                   ha='center', va=va_rand, fontsize=7, color=COLORS['random'])
        
        # 标注 Greedy 值 (仅第一组)
        va_gred = 'bottom' if gred_val >= 0 else 'top'
        off_gred = 3 if gred_val >= 0 else -3
        ax.annotate(f'{gred_val:.1f}',
                   xy=(bars_gred[0].get_x() + bars_gred[0].get_width()/2, gred_val),
                   xytext=(0, off_gred), textcoords="offset points",
                   ha='center', va=va_gred, fontsize=7, color=COLORS['greedy'])
        
        ax.set_xlabel('Episode Range')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(window_labels, fontsize=9)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.suptitle('Algorithm Performance Comparison (per 100 Episodes)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    filepath = os.path.join(save_dir, f'algorithm_comparison_bars_{timestamp}.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 柱状对比图已保存至: {filepath}")
