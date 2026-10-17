"""
dqn_baseline.py — 轻量 DQN 基线 (与 PPO 共用同一环境接口)

包含:
  - DQNNetwork:  简单 Q 网络 (MLP, 256->128)
  - ReplayBuffer: 经验回放缓冲区
  - DQNAgent:    epsilon-greedy + 目标网络 + Bellman 更新
  - run_dqn_training: DQN 训练循环 (与 PPO 相同的环境交互方式)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional, Dict

from config import (
    NUM_EPISODES, STEPS_PER_EPISODE, MAX_BEAMS, E_AVG, LAMBDA_ARRIVAL,
    F_MIN_NUM, F_MID_NUM, F_MAX_NUM,
)
from env import UAV_HAPMECEnv, get_state_dim

# ==================== DQN 超参数 ====================
DQN_LR                 = 1e-3     # 学习率 (比 PPO 稍高, 加速收敛)
DQN_GAMMA              = 0.99     # 折扣因子 (与 PPO 一致)
DQN_EPS_START          = 1.0      # 初始探索率
DQN_EPS_END            = 0.05     # 最终探索率
DQN_EPS_DECAY          = 0.995    # 每 episode 衰减因子
DQN_BUFFER_SIZE        = 20000    # 回放缓冲区容量
DQN_BATCH_SIZE         = 128      # 批次大小 (与 PPO 一致)
DQN_TARGET_UPDATE_FREQ = 200      # 目标网络同步间隔 (步)
DQN_UPDATE_FREQ        = 4        # 更新间隔 (步)
DQN_HIDDEN_DIM         = 256      # 隐藏层维度


class DQNNetwork(nn.Module):
    """轻量 Q 网络: state → Q(s, a)  for all a"""
    
    def __init__(self, state_dim: int, num_actions: int, 
                 hidden_dim: int = DQN_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = DQN_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """DQN 智能体: ε-greedy 探索 + 目标网络 + 经验回放"""
    
    def __init__(self, state_dim: int, num_actions: int, device: str = 'cpu'):
        self.state_dim   = state_dim
        self.num_actions = num_actions
        self.device      = device
        
        # Q 网络 & 目标网络
        self.q_net      = DQNNetwork(state_dim, num_actions).to(device)
        self.target_net = DQNNetwork(state_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        
        self.optimizer     = optim.Adam(self.q_net.parameters(), lr=DQN_LR)
        self.replay_buffer = ReplayBuffer()
        
        self.epsilon    = DQN_EPS_START
        self.step_count = 0
    
    # ---- 动作选择 ----
    def get_action(self, state: np.ndarray, 
                   action_mask: np.ndarray = None) -> int:
        """ε-greedy 动作选择, 支持动作掩码 (不可用动作不会被选中)"""
        if np.random.random() < self.epsilon:
            # 随机探索 (仅合法动作)
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                return int(np.random.choice(valid)) if len(valid) > 0 else 0
            return np.random.randint(self.num_actions)
        
        # 贪心选择
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t).squeeze(0)
        
        if action_mask is not None:
            mask_t = torch.BoolTensor(action_mask).to(self.device)
            q_values[~mask_t] = -float('inf')
        
        return int(q_values.argmax().item())
    
    # ---- Bellman 更新 ----
    def update(self) -> Optional[float]:
        """执行一次 DQN 更新, 返回 loss 值 (样本不足时返回 None)"""
        if len(self.replay_buffer) < DQN_BATCH_SIZE:
            return None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(DQN_BATCH_SIZE)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)
        
        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # target = r + γ * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q  = self.target_net(next_states).max(1)[0]
            target  = rewards + DQN_GAMMA * next_q * (1.0 - dones)
        
        loss = F.mse_loss(q_values, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        
        return loss.item()
    
    # ---- 辅助 ----
    def decay_epsilon(self):
        """每 episode 衰减探索率"""
        self.epsilon = max(DQN_EPS_END, self.epsilon * DQN_EPS_DECAY)
    
    def sync_target(self):
        """同步目标网络"""
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def save(self, filepath: str):
        torch.save({
            'q_net':      self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict(),
            'epsilon':    self.epsilon,
        }, filepath)
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']


def run_dqn_training(
    num_episodes: int      = NUM_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
    save_dir: str          = "./results",
    device: str            = 'cuda' if torch.cuda.is_available() else 'cpu',
    low_uavs: int          = None,
    mid_uavs: int          = None,
    max_uavs: int          = None,
    e_avg: float           = None,
    task_arrive_rate: float = None,
) -> Dict:
    """
    DQN 训练循环 — 与 PPO 相同的环境接口 & 数据采集
    
    Args:
        num_episodes:     训练 episode 数 (应与 PPO 一致)
        steps_per_episode: 每 episode 步数 (应与 PPO 一致)
        save_dir:         模型保存目录
        device:           计算设备
        low_uavs:         低算力 UAV 数量 (默认 None=使用 config 默认值)
        mid_uavs:         中算力 UAV 数量
        max_uavs:         高算力 UAV 数量
        e_avg:            能耗上限 (默认 None=使用 config 默认值)
        task_arrive_rate: 任务到达率 GFLOPS/s (默认 None=使用 config 默认值)
    
    Returns:
        dict: 训练历史 (与 PPO history 结构兼容)
              {episode_rewards, episode_delays, episode_energies,
               episode_queue_lengths, lyapunov_Z, lyapunov_Y}
    """
    import os
    from datetime import datetime

    # ---- 计算 UAV 配置 (默认 5:3:2, 与 env 构造函数默认值一致) ----
    if low_uavs is None:
        low_uavs = F_MIN_NUM
    if mid_uavs is None:
        mid_uavs = F_MID_NUM
    if max_uavs is None:
        max_uavs = F_MAX_NUM
    total_uavs = low_uavs + mid_uavs + max_uavs
    if e_avg is None:
        e_avg = E_AVG * (total_uavs / 10.0)
    if task_arrive_rate is None:
        task_arrive_rate = LAMBDA_ARRIVAL
    
    print(f"\n[INFO] ========== DQN Baseline 训练 ==========")
    print(f"[INFO] 使用设备: {device}")
    print(f"[INFO] UAV 配置: {low_uavs}低 + {mid_uavs}中 + {max_uavs}高 = {total_uavs} 架")
    print(f"[INFO] 任务到达率: {task_arrive_rate} GFLOPS/s")
    print(f"[INFO] Episode 数: {num_episodes}, "
          f"每 Episode 步数: {steps_per_episode}")
    
    env = UAV_HAPMECEnv(low_uavs=low_uavs, mid_uavs=mid_uavs, max_uavs=max_uavs,
                        task_arrive_rate=task_arrive_rate, e_avg=e_avg)
    state_dim   = get_state_dim()
    num_actions = env.num_actions
    num_uavs    = env.num_uavs
    
    agent = DQNAgent(state_dim, num_actions, device)
    
    # ---- 历史记录 ----
    episode_rewards       = []
    episode_delays        = []
    episode_energies      = []
    episode_queue_lengths = []
    lyapunov_Z_history    = []
    lyapunov_Y_history    = []
    
    global_step = 0
    
    # ---- 训练循环 ----
    for ep in range(num_episodes):
        env.reset()
        
        ep_rewards  = []
        ep_delays   = []
        ep_energies = []
        
        for step in range(steps_per_episode):
            step_experiences = []  # 暂存本时隙经验
            
            # ---- 每个 UAV 依次决策 ----
            for uav_id in range(num_uavs):
                state       = env._get_state_vector(uav_id)
                action_mask = env.get_action_mask(uav_id)
                
                action     = agent.get_action(state, action_mask)
                next_state, _, _, _, _ = env.step(uav_id, action)
                
                step_experiences.append({
                    'uav_id':     uav_id,
                    'state':      state,
                    'action':     action,
                    'next_state': next_state,
                })
            
            # ---- 时隙结束: 结算奖励 ----
            rewards_dict = env.end_of_step()
            
            # 收集本时隙完成任务的延迟/能耗 (仅刚完成的任务)
            ep_delays.extend(env._step_completed_delays)
            ep_energies.extend(env._step_completed_energies)
            
            # ---- 回填奖励 → replay buffer ----
            for exp in step_experiences:
                uid    = exp['uav_id']
                reward = rewards_dict.get(uid, 0.0)
                agent.replay_buffer.push(
                    exp['state'], exp['action'], reward,
                    exp['next_state'], False
                )
                ep_rewards.append(reward)
            
            global_step += 1
            
            # ---- 定期 DQN 更新 ----
            if global_step % DQN_UPDATE_FREQ == 0:
                agent.update()
            
            # ---- 同步目标网络 ----
            if global_step % DQN_TARGET_UPDATE_FREQ == 0:
                agent.sync_target()
        
        # ---- Episode 统计 ----
        avg_reward = np.mean(ep_rewards)  if ep_rewards  else 0.0
        avg_delay  = np.mean(ep_delays)   if ep_delays   else 0.0
        avg_energy = np.mean(ep_energies) if ep_energies else 0.0
        
        episode_rewards.append(avg_reward)
        episode_delays.append(avg_delay)
        episode_energies.append(avg_energy)
        
        Q_t, Z_t, Y_t = env.get_lyapunov_state()
        episode_queue_lengths.append(Q_t)
        lyapunov_Z_history.append(Z_t)
        lyapunov_Y_history.append(Y_t)
        
        # 衰减 ε
        agent.decay_epsilon()
        
        # 打印进度
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[DQN Ep {ep+1:4d}/{num_episodes}] "
                  f"Reward={avg_reward:+.3f}  Delay={avg_delay:.1f}s  "
                  f"Energy={avg_energy:.1f}J  ε={agent.epsilon:.3f}")
    
    print(f"[INFO] DQN 训练完成! 最终 ε={agent.epsilon:.4f}")
    
    # 保存 DQN 模型
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agent.save(os.path.join(save_dir, f"dqn_model_{timestamp}.pt"))
    
    return {
        'episode_rewards':       episode_rewards,
        'episode_delays':        episode_delays,
        'episode_energies':      episode_energies,
        'episode_queue_lengths': episode_queue_lengths,
        'lyapunov_Z':            lyapunov_Z_history,
        'lyapunov_Y':            lyapunov_Y_history,
    }
