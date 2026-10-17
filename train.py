"""
train.py — Lyapunov 整形奖励 + PPO 训练主循环

训练流程:
  1. 初始化环境、PPO 网络、均衡采样缓冲区
  2. 每个时隙: 所有 UAV 依次决策 -> 收集经验 (含动作掩码)
  3. 时隙结束: 结算中继 + Lyapunov 奖励
  4. 定期触发 PPO 策略更新
  5. 动态调整超参数 (学习率线性衰减、熵系数退火)
"""

import numpy as np
import torch
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Tuple
from collections import defaultdict

from config import *
from env import UAV_HAPMECEnv, get_state_dim
from ppo_trainer import PPOTrainer
from dqn_baseline import run_dqn_training
from plot_bars import plot_comparison_bar_charts
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt


def set_seed(seed: int = 42):
    """设置所有随机种子以确保可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    num_episodes: int = NUM_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
    update_interval: int = UPDATE_INTERVAL,
    save_dir: str = "./results",
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    run_baselines: bool = False,
    run_dqn: bool = False,
    low_uavs: int = None,
    mid_uavs: int = None,
    max_uavs: int = None,
    task_arrive_rate: float = None,
):
    """
    PPO 训练主循环

    Args:
        num_episodes:     训练 episode 数
        steps_per_episode: 每个 episode 的时隙数
        update_interval:   经验收集多少步后触发 PPO 更新
        save_dir:          结果保存目录
        device:            计算设备
        run_baselines:     是否在训练后运行贪心/随机基准对比
        run_dqn:           是否在训练后运行 DQN 基线训练
        low_uavs:          低算力 UAV 数量 (默认 None=使用 config 默认值)
        mid_uavs:          中算力 UAV 数量
        max_uavs:          高算力 UAV 数量
        task_arrive_rate:  任务到达率 GFLOPS/s (默认 None=使用 config 默认值)
    """
    # ---- 计算 UAV 配置 (默认 5:3:2, 与 env 构造函数默认值一致) ----
    if low_uavs is None:
        low_uavs = F_MIN_NUM
    if mid_uavs is None:
        mid_uavs = F_MID_NUM
    if max_uavs is None:
        max_uavs = F_MAX_NUM
    total_uavs = low_uavs + mid_uavs + max_uavs
    # E_AVG 按 UAV 数量等比缩放 (基准: 10 UAVs → 30000 J/slot)
    scaled_e_avg = E_AVG * (total_uavs / 10.0)
    
    # 任务到达率 (默认使用 config 中的 LAMBDA_ARRIVAL)
    if task_arrive_rate is None:
        task_arrive_rate = LAMBDA_ARRIVAL
    
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] 初始化 PPO 训练...")
    print(f"       UAV 配置: {low_uavs}低 + {mid_uavs}中 + {max_uavs}高 = {total_uavs} 架")
    print(f"       任务到达率: {task_arrive_rate} GFLOPS/s")
    print(f"       Episode 数: {num_episodes}")
    print(f"       每 Episode 步数: {steps_per_episode}")
    print(f"       更新间隔: {update_interval}")
    print(f"       E_AVG 缩放: {scaled_e_avg:.0f} J/slot")
    
    # ---- 创建保存目录 ----
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # ---- 初始化环境与训练器 ----
    env = UAV_HAPMECEnv(low_uavs=low_uavs, mid_uavs=mid_uavs, max_uavs=max_uavs,
                        task_arrive_rate=task_arrive_rate, e_avg=scaled_e_avg)
    state_dim = get_state_dim()
    trainer = PPOTrainer(state_dim=state_dim, num_uavs=env.num_uavs, device=device)
    
    num_uavs = env.num_uavs
    
    # ---- 训练统计 ----
    episode_rewards = []          # 每个 episode 平均奖励
    episode_delays = []           # 每个 episode 平均时延
    episode_energies = []         # 每个 episode 平均能耗
    episode_queue_lengths = []    # 每个 episode Q(t) 均值
    lyapunov_Z_history = []       # Z(t) 历史
    lyapunov_Y_history = []       # Y(t) 历史
    
    global_step = 0
    best_avg_reward = -float('inf')
    
    # ---- 训练循环 ----
    for ep in range(num_episodes):
        obs, _ = env.reset()
        
        ep_rewards = []
        ep_delays = []
        ep_energies = []
        ep_Q_vals = []
        
        # 动态调整学习率 (线性衰减, 不低于 PPO_LR_MIN)
        lr_now = max(PPO_LR_MIN, PPO_LR * (1.0 - ep / max(1, num_episodes)))
        trainer.set_lr(lr_now)
        
        # 熵退火: 初期高探索 → 后期低探索 (保留少量随机性)
        progress = ep / max(1, num_episodes)
        ent_coef = PPO_ENTROPY_COEF * (1.0 - progress) + PPO_ENTROPY_COEF_END * progress
        trainer.set_entropy_coef(ent_coef)
        
        for step in range(steps_per_episode):
            # 存储本时隙的经验 (等 end_of_step 后回填奖励)
            step_experiences = []  # [(uav_id, state, action, log_prob, value)]
            
            # ---- 每个 UAV 依次决策 (多时隙模型: 忙碌 UAV 跳过) ----
            for uav_id in range(num_uavs):
                uav = env.uavs[uav_id]
                state = env._get_state_vector(uav_id)
                action_mask = env.get_action_mask(uav_id)

                if uav.state != 0:
                    # 忙碌 UAV: 使用占位动作, step() 自动跳过
                    action = 0  # dummy
                    log_prob, value = 0.0, 0.0
                else:
                    # 空闲 UAV: PPO 决策
                    if not action_mask.any():
                        action = 0  # 无可用波束, 默认机载
                        log_prob, value = 0.0, 0.0
                    else:
                        action, log_prob, value = trainer.get_action(state, uav_id, action_mask)

                # 执行动作 (忙碌 UAV 会在 step 中自动跳过)
                next_obs, _, terminated, truncated, info = env.step(uav_id, action)

                # 仅为空闲 UAV 暂存经验 (掩码随经验存储, 供更新时复算)
                if not info.get('busy', False):
                    step_experiences.append({
                        'uav_id': uav_id,
                        'state': state,
                        'action': action,
                        'log_prob': log_prob,
                        'value': value,
                        'next_state': next_obs,
                        'mask': action_mask,
                    })
            
            # ---- 时隙结束: 结算奖励 ----
            rewards_dict = env.end_of_step()
            
            # 收集本时隙完成任务的延迟/能耗 (仅收集刚完成的任务, 避免重复采样)
            ep_delays.extend(env._step_completed_delays)
            ep_energies.extend(env._step_completed_energies)
            
            # ---- 回填奖励并存储经验 ----
            for exp in step_experiences:
                uid = exp['uav_id']
                reward = rewards_dict.get(uid, 0.0)
                trainer.store_experience(
                    uav_id=uid,
                    state=exp['state'],
                    action=exp['action'],
                    reward=reward,
                    next_state=exp['next_state'],
                    done=False,
                    log_prob=exp['log_prob'],
                    value=exp['value'],
                    action_mask=exp['mask'],
                )
                ep_rewards.append(reward)
            
            ep_Q_vals.append(env.Q)
            
            global_step += 1
            
            # ---- 定期 PPO 更新 ----
            if global_step % update_interval == 0 and trainer.is_update_ready():
                loss_info = trainer.update()
                if loss_info:
                    print(f"  [PPO Update #{trainer.update_count}] "
                          f"policy_loss={loss_info['policy_loss']:.4f}, "
                          f"critic_loss={loss_info['critic_loss']:.4f}, "
                          f"entropy={loss_info['entropy']:.4f}")
        
        # ---- Episode 统计 ----
        avg_reward = np.mean(ep_rewards) if ep_rewards else 0.0
        avg_delay = np.mean(ep_delays) if ep_delays else 0.0
        avg_energy = np.mean(ep_energies) if ep_energies else 0.0
        avg_Q = np.mean(ep_Q_vals) if ep_Q_vals else 0.0
        
        episode_rewards.append(avg_reward)
        episode_delays.append(avg_delay)
        episode_energies.append(avg_energy)
        episode_queue_lengths.append(avg_Q)
        
        Q_t, Z_t, Y_t = env.get_lyapunov_state()
        lyapunov_Z_history.append(Z_t)
        lyapunov_Y_history.append(Y_t)
        
        # 打印进度
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[Ep {ep+1:4d}/{num_episodes}] "
                  f"Reward={avg_reward:+.3f}  Delay={avg_delay:.1f}s  "
                  f"Energy={avg_energy:.1f}J  Q={avg_Q:.1f}  "
                  f"Z={Z_t:.1f}  Y={Y_t:.1f}  LR={lr_now:.2e}")
        
        # 保存最佳模型
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            trainer.save(os.path.join(save_dir, f"best_model_{timestamp}.pt"))
    
    # ---- 训练结束 ----
    print(f"\n[INFO] 训练完成! 最佳平均奖励: {best_avg_reward:.3f}")
    
    # 保存最终模型
    trainer.save(os.path.join(save_dir, f"final_model_{timestamp}.pt"))
    
    # 保存训练历史 (不含 baseline)
    history = {
        'episode_rewards': episode_rewards,
        'episode_delays': episode_delays,
        'episode_energies': episode_energies,
        'episode_queue_lengths': episode_queue_lengths,
        'lyapunov_Z': lyapunov_Z_history,
        'lyapunov_Y': lyapunov_Y_history,
        'best_reward': best_avg_reward,
    }
    
    # ---- 运行基准对比 (可选): 随机策略 + 贪心策略 ----
    baseline_results = None
    if run_baselines:
        print("\n" + "=" * 60)
        print("  运行基准对比 (随机策略 vs 贪心策略)")
        print("=" * 60)
        baseline_results = run_baseline_comparison(
            env, num_episodes=num_episodes, steps_per_episode=steps_per_episode
        )
        history['baseline'] = baseline_results
    
    # ---- 运行 DQN Baseline (可选): 收敛算法对比 ----
    dqn_history = None
    if run_dqn:
        dqn_history = run_dqn_training(
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            save_dir=save_dir,
            device=device,
            low_uavs=low_uavs,
            mid_uavs=mid_uavs,
            max_uavs=max_uavs,
            e_avg=scaled_e_avg,
            task_arrive_rate=task_arrive_rate,
        )
        history['dqn'] = dqn_history
    
    with open(os.path.join(save_dir, f"training_history_{timestamp}.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    # 打印对比摘要
    if baseline_results is not None or dqn_history is not None:
        _print_baseline_summary(history, baseline_results, dqn_history)
    
    # 绘制训练曲线 (自动检测 history 中的 DQN / baseline 数据)
    plot_training_curves(history, save_dir, timestamp)
    
    # 绘制柱状对比图 (需要 baseline 数据)
    if baseline_results is not None:
        plot_comparison_bar_charts(
            ppo_history=history,
            baseline_results=baseline_results,
            dqn_history=dqn_history,
            save_dir=save_dir,
        )
    
    return history


def run_baseline_comparison(env: UAV_HAPMECEnv, num_episodes: int = 50,
                            steps_per_episode: int = 50):
    """
    运行随机策略和贪心策略作为对比基准
    
    Args:
        env:              已初始化的环境实例
        num_episodes:     运行 episode 数 (应与 PPO 训练一致)
        steps_per_episode: 每 episode 步数 (应与 PPO 训练一致)
    
    Returns:
        dict: {'random': {...}, 'greedy': {...}}
              每个策略包含 rewards, delays, energies, Q, Z, Y 的历史序列
    """
    num_uavs = env.num_uavs
    
    def run_policy(name: str, policy_fn) -> Dict:
        rewards_hist = []
        delays_hist = []
        energies_hist = []
        Q_hist = []
        Z_hist = []
        Y_hist = []
        
        for ep in range(num_episodes):
            env.reset()
            ep_rewards = []
            ep_delays = []
            ep_energies = []
            
            for step in range(steps_per_episode):
                for uav_id in range(num_uavs):
                    uav = env.uavs[uav_id]
                    if uav.state != 0:
                        # 忙碌 UAV: dummy 动作
                        env.step(uav_id, 0)
                    else:
                        state = env._get_state_vector(uav_id)
                        mask = env.get_action_mask(uav_id)
                        action = policy_fn(state, mask, uav_id, env)
                        env.step(uav_id, action)
                
                rewards_dict = env.end_of_step()
                for uid in range(num_uavs):
                    ep_rewards.append(rewards_dict.get(uid, 0.0))
                # 收集本时隙完成任务的延迟/能耗 (仅刚完成的任务)
                ep_delays.extend(env._step_completed_delays)
                ep_energies.extend(env._step_completed_energies)
            
            rewards_hist.append(np.mean(ep_rewards))
            delays_hist.append(np.mean(ep_delays))
            energies_hist.append(np.mean(ep_energies))
            Q, Z, Y = env.get_lyapunov_state()
            Q_hist.append(Q)
            Z_hist.append(Z)
            Y_hist.append(Y)
        
        return {
            'rewards': rewards_hist,
            'delays': delays_hist,
            'energies': energies_hist,
            'Q': Q_hist,
            'Z': Z_hist,
            'Y': Y_hist,
        }
    
    # 随机策略
    def random_policy(state, mask, uav_id, env):
        valid = np.where(mask)[0]
        return np.random.choice(valid) if len(valid) > 0 else 0
    
    # 贪心策略: 选择最近的簇; 无簇则机载计算 (HAP 自动选本地/中继)
    def greedy_policy(state, mask, uav_id, env):
        valid = np.where(mask)[0]
        cluster_actions = [a for a in valid if 1 <= a <= NUM_BEAMS]
        if not cluster_actions:
            return 0  # 机载计算
        
        # 找最近的簇: 计算 UAV 到每个簇中心的距离
        uav = env.uavs[uav_id]
        best_action = cluster_actions[0]
        best_dist = float('inf')
        for a in cluster_actions:
            cluster_idx = a - 1
            if cluster_idx < len(env.HAP.cluster_queue):
                c = env.HAP.cluster_queue[cluster_idx]
                dist = np.hypot(uav.pos[0] - c['center_x'], uav.pos[1] - c['center_y'])
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
        return best_action
    
    print("\n[INFO] 运行随机策略基准...")
    random_results = run_policy('random', random_policy)
    print("[INFO] 运行贪心策略基准...")
    greedy_results = run_policy('greedy', greedy_policy)
    
    return {'random': random_results, 'greedy': greedy_results}


def plot_training_curves(history: Dict, save_dir: str, timestamp: str,
                         baseline_results: dict = None):
    """
    绘制训练曲线 (4 子图)
    优先显示 DQN (收敛算法) 作为对比曲线;
    若无 DQN 则回退到随机/贪心 baseline
    
    Args:
        history:          训练历史字典 (可含 'dqn' / 'baseline' 键)
        save_dir:         保存目录
        timestamp:        时间戳
        baseline_results: 基准结果 (若为 None 则从 history 读取)
    """
    # ---- 检测对比数据: DQN 优先 (收敛算法) ----
    dqn_history = history.get('dqn', None)
    has_dqn = dqn_history is not None
    
    if baseline_results is None:
        baseline_results = history.get('baseline', None)
    # 有 DQN 时训练曲线不需要随机/贪心 (它们不收敛, 干扰视觉)
    has_bl = (baseline_results is not None) and not has_dqn
    
    bl_colors = {'random': '#999999', 'greedy': '#F39C12'}
    bl_labels = {'random': 'Random Policy', 'greedy': 'Greedy Policy'}
    dqn_color = "#02F7FF"
    dqn_label = 'DQN Baseline'
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = np.arange(1, len(history['episode_rewards']) + 1)
    
    # ---- 子图 1: 平均奖励 ----
    ax = axes[0, 0]
    ax.plot(episodes, history['episode_rewards'], 'b-', linewidth=1.5,
            alpha=0.8, label='PPO')
    if has_dqn:
        dqn_ep = np.arange(1, len(dqn_history['episode_rewards']) + 1)
        ax.plot(dqn_ep, dqn_history['episode_rewards'],
                color=dqn_color, linestyle='-', linewidth=1.5,
                alpha=0.8, label=dqn_label)
    elif has_bl:
        for name in ['random', 'greedy']:
            if name in baseline_results:
                bl_ep = np.arange(1, len(baseline_results[name]['rewards']) + 1)
                ax.plot(bl_ep, baseline_results[name]['rewards'],
                        color=bl_colors[name], linestyle='-', linewidth=1.2,
                        alpha=0.7, label=bl_labels[name])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward')
    ax.set_title('PPO Training: Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ---- 子图 2: 平均时延 ----
    ax = axes[0, 1]
    ax.plot(episodes, history['episode_delays'], 'r-', linewidth=1.5,
            alpha=0.8, label='PPO')
    if has_dqn:
        dqn_ep = np.arange(1, len(dqn_history['episode_delays']) + 1)
        ax.plot(dqn_ep, dqn_history['episode_delays'],
                color=dqn_color, linestyle='-', linewidth=1.5,
                alpha=0.8, label=dqn_label)
    elif has_bl:
        for name in ['random', 'greedy']:
            if name in baseline_results:
                bl_ep = np.arange(1, len(baseline_results[name]['delays']) + 1)
                ax.plot(bl_ep, baseline_results[name]['delays'],
                        color=bl_colors[name], linestyle='-', linewidth=1.2,
                        alpha=0.7, label=bl_labels[name])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Delay (s)')
    ax.set_title('Average Delay per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ---- 子图 3: 平均能耗 ----
    ax = axes[1, 0]
    ax.plot(episodes, history['episode_energies'], 'g-', linewidth=1.5,
            alpha=0.8, label='PPO')
    if has_dqn:
        dqn_ep = np.arange(1, len(dqn_history['episode_energies']) + 1)
        ax.plot(dqn_ep, dqn_history['episode_energies'],
                color=dqn_color, linestyle='-', linewidth=1.5,
                alpha=0.8, label=dqn_label)
    elif has_bl:
        for name in ['random', 'greedy']:
            if name in baseline_results:
                bl_ep = np.arange(1, len(baseline_results[name]['energies']) + 1)
                ax.plot(bl_ep, baseline_results[name]['energies'],
                        color=bl_colors[name], linestyle='-', linewidth=1.2,
                        alpha=0.7, label=bl_labels[name])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Energy (J)')
    ax.set_title('Average Energy per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ---- 子图 4: Lyapunov 队列稳定性 ----
    ax = axes[1, 1]
    ax.plot(episodes, history['lyapunov_Z'], 'orange', linewidth=1.2,
            alpha=0.8, label='PPO Z(t) Energy')
    ax.plot(episodes, history['lyapunov_Y'], 'purple', linewidth=1.2,
            alpha=0.8, label='PPO Y(t) Rate')
    if has_dqn:
        dqn_ep = np.arange(1, len(dqn_history['lyapunov_Z']) + 1)
        ax.plot(dqn_ep, dqn_history['lyapunov_Z'],
                color=dqn_color, linestyle='-', linewidth=1.0,
                alpha=0.6, label=f'{dqn_label} Z(t)')
        ax.plot(dqn_ep, dqn_history['lyapunov_Y'],
                color=dqn_color, linestyle='--', linewidth=1.0,
                alpha=0.6, label=f'{dqn_label} Y(t)')
    elif has_bl:
        for name in ['random', 'greedy']:
            if name in baseline_results and 'Z' in baseline_results[name]:
                bl_ep = np.arange(1, len(baseline_results[name]['Z']) + 1)
                ax.plot(bl_ep, baseline_results[name]['Z'],
                        color=bl_colors[name], linestyle='-', linewidth=1.0,
                        alpha=0.5, label=f'{bl_labels[name]} Z(t)')
                ax.plot(bl_ep, baseline_results[name]['Y'],
                        color=bl_colors[name], linestyle='-', linewidth=1.0,
                        alpha=0.5, label=f'{bl_labels[name]} Y(t)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Queue Value')
    ax.set_title('Lyapunov Virtual Queue Stability')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(save_dir, f"training_curves_{timestamp}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"[INFO] 训练曲线已保存至: {filepath}")


def _print_baseline_summary(history: Dict, baseline_results: Dict = None,
                            dqn_history: Dict = None):
    """打印 PPO vs Baseline 策略性能对比摘要 (含 DQN)"""
    bl_labels = {'random': 'Random Policy', 'greedy': 'Greedy Policy'}
    print("\n" + "=" * 60)
    print("  策略性能对比摘要")
    print("=" * 60)
    print(f"{'Strategy':<24} {'Reward':>10} {'Delay(s)':>12} {'Energy(J)':>12}")
    print("-" * 60)
    
    # 随机 & 贪心
    if baseline_results is not None:
        for name in ['random', 'greedy']:
            if name in baseline_results:
                r = baseline_results[name]['rewards']
                d = baseline_results[name]['delays']
                e = baseline_results[name]['energies']
                print(f"{bl_labels[name]:<24} {np.mean(r):>+10.2f} "
                      f"{np.mean(d):>12.1f} {np.mean(e):>12.1f}")
    
    # DQN (收敛后 50%)
    if dqn_history is not None:
        half_dqn = len(dqn_history['episode_rewards']) // 2
        dqn_r = np.mean(dqn_history['episode_rewards'][half_dqn:]) if half_dqn > 0 else np.mean(dqn_history['episode_rewards'])
        dqn_d = np.mean(dqn_history['episode_delays'][half_dqn:]) if half_dqn > 0 else np.mean(dqn_history['episode_delays'])
        dqn_e = np.mean(dqn_history['episode_energies'][half_dqn:]) if half_dqn > 0 else np.mean(dqn_history['episode_energies'])
        print(f"{'DQN Baseline':<24} {dqn_r:>+10.2f} {dqn_d:>12.1f} {dqn_e:>12.1f}")
    
    # PPO 收敛后性能 (后 50%)
    half = len(history['episode_rewards']) // 2
    ppo_r = np.mean(history['episode_rewards'][half:]) if half > 0 else np.mean(history['episode_rewards'])
    ppo_d = np.mean(history['episode_delays'][half:]) if half > 0 else np.mean(history['episode_delays'])
    ppo_e = np.mean(history['episode_energies'][half:]) if half > 0 else np.mean(history['episode_energies'])
    print(f"{'PPO':<24} {ppo_r:>+10.2f} {ppo_d:>12.1f} {ppo_e:>12.1f}")
    print("-" * 60)


# ==================== 入口 ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PPO training for UAV-HAP MEC offloading")
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES,
                        help=f'训练 episode 数 (默认: {NUM_EPISODES})')
    parser.add_argument('--steps', type=int, default=STEPS_PER_EPISODE,
                        help=f'每 episode 步数 (默认: {STEPS_PER_EPISODE})')
    parser.add_argument('--update', type=int, default=UPDATE_INTERVAL,
                        help=f'PPO 更新间隔 (默认: {UPDATE_INTERVAL})')
    parser.add_argument('--save', type=str, default='./results',
                        help='结果保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--uavs', type=str, default=None,
                        help='三档算力 UAV 数量, 形如 "5,3,2" (默认: config 的 5,3,2)')
    parser.add_argument('--arrival', type=float, default=None,
                        help=f'任务到达率 GFLOPS/s (默认: {LAMBDA_ARRIVAL})')
    parser.add_argument('--baseline', action='store_true', default=False,
                        help='启用贪心策略与随机策略基准对比 (默认关闭)')
    parser.add_argument('--dqn', action='store_true', default=False,
                        help='启用 DQN 基线对比训练 (默认关闭)')

    args = parser.parse_args()

    set_seed(args.seed)

    low_uavs = mid_uavs = max_uavs = None
    if args.uavs:
        low_uavs, mid_uavs, max_uavs = (int(x) for x in args.uavs.split(','))

    print("=" * 60)
    print("  UAV-HAP 边缘计算卸载仿真 (Lyapunov + PPO)")
    print("=" * 60)
    if args.dqn:
        print("  [MODE] PPO + DQN 收敛对比")
    if args.baseline:
        print("  [MODE] 含随机/贪心基准 + 柱状图")

    # 主训练 (含可选基准对比)
    history = run_training(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        update_interval=args.update,
        save_dir=args.save,
        run_baselines=args.baseline,
        run_dqn=args.dqn,
        low_uavs=low_uavs,
        mid_uavs=mid_uavs,
        max_uavs=max_uavs,
        task_arrive_rate=args.arrival,
    )
    
    print("\n[DONE] 仿真完成!")
