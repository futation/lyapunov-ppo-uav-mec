"""ppo_trainer.py — PPO 训练器 (GAE + Clip 目标 + 熵正则)

实现完整的多智能体 PPO 策略更新流程:
广义优势估计 (GAE)、PPO Clip 目标、Critic 损失、熵正则化、动作掩码。

掩码一致性: 采样与更新均使用动作掩码 (buffer 存储每条经验的掩码),
保证新旧策略的对数概率在同一合法动作集合下计算, ratio 才有意义。

已知简化: GAE 递推是在打乱后的批次上进行的 (样本来自不同 UAV、
不同时隙), 等价于一种带折扣的混合优势估计; 若需严格按轨迹计算,
可按 (UAV, episode) 分组重写 update() 中的递推顺序。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional, Dict

from config import (
    PPO_LR, PPO_EPS_CLIP, PPO_GAMMA, PPO_LAMBDA_GAE,
    PPO_ENTROPY_COEF, PPO_CRITIC_COEF, PPO_BATCH_SIZE,
    PPO_EPOCHS, PPO_MAX_GRAD_NORM
)
from networks import ActorNetwork, CriticNetwork
from balanced_buffer import PerUAVBalancedBuffer


class PPOTrainer:
    """多智能体 PPO 训练器: 管理共享 Actor/Critic、均衡缓冲区和策略更新"""

    def __init__(self,
                 state_dim: int = None,
                 num_uavs: int = None,
                 device: str = 'cpu',
                 lr: float = PPO_LR):
        """
        Args:
            state_dim: 状态向量维度 (None 则从 env 推断)
            num_uavs:  无人机数量
            device:    训练设备
            lr:        学习率
        """
        if state_dim is None:
            from env import get_state_dim
            state_dim = get_state_dim()
        if num_uavs is None:
            from config import NUM_UAVS
            num_uavs = NUM_UAVS

        self.num_uavs = num_uavs
        self.state_dim = state_dim
        self.device = device

        # 初始化网络
        self.actor = ActorNetwork(self.state_dim, num_uavs=self.num_uavs).to(device)
        self.critic = CriticNetwork(self.state_dim, num_uavs=self.num_uavs).to(device)

        # 优化器
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr},
        ])

        # 按 UAV 均衡采样的经验缓冲区
        self.buffer = PerUAVBalancedBuffer(num_uavs=self.num_uavs, device=device)

        # 可动态调整的超参数
        self.entropy_coef = PPO_ENTROPY_COEF

        # 训练统计
        self.update_count = 0
        self.total_samples_collected = 0

    def set_entropy_coef(self, coef: float):
        """动态调整熵正则系数 (用于退火)"""
        self.entropy_coef = coef

    def store_experience(self, uav_id: int,
                         state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool,
                         log_prob: float, value: float,
                         action_mask: np.ndarray = None):
        """存储一条经验 (含采样时的动作掩码) 到缓冲区"""
        self.buffer.add(uav_id, state, action, reward, next_state, done,
                        log_prob, value, action_mask=action_mask)
        self.total_samples_collected += 1

    def is_update_ready(self) -> bool:
        """检查缓冲区是否有足够样本触发 PPO 更新"""
        return self.buffer.is_ready()

    def update(self) -> Optional[Dict[str, float]]:
        """执行一次 PPO 策略更新, 缓冲区不足时返回 None"""
        batch = self.buffer.sample()
        if batch is None:
            return None

        states        = batch['states']
        uav_ids       = batch['uav_ids']
        actions       = batch['actions']
        rewards       = batch['rewards']
        next_states   = batch['next_states']
        dones         = batch['dones']
        old_log_probs = batch['old_log_probs']
        old_values    = batch['old_values']
        action_masks  = batch['action_masks']

        batch_size_actual = states.shape[0]

        # ---- 1. GAE 优势估计 ----
        with torch.no_grad():
            next_values = self.critic(next_states, uav_ids).squeeze(-1)

        advantages = torch.zeros(batch_size_actual, device=self.device)
        returns    = torch.zeros(batch_size_actual, device=self.device)

        gae = 0.0
        for t in reversed(range(batch_size_actual)):
            # TD 误差: delta_t = r_t + gamma * V(s_t+1) * (1-done) - V(s_t)
            next_val = next_values[t] if not dones[t] else 0.0
            delta = rewards[t] + PPO_GAMMA * next_val - old_values[t]
            # GAE: A_t = delta_t + gamma * lambda * A_t+1
            gae = delta + PPO_GAMMA * PPO_LAMBDA_GAE * gae * (1 - float(dones[t]))
            advantages[t] = gae
            returns[t] = advantages[t] + old_values[t]

        # 标准化优势 (稳定训练)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 2. PPO 多 epoch 更新 ----
        total_policy_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0

        for _ in range(PPO_EPOCHS):
            # 重新评估动作 (使用与采样时一致的掩码)
            new_log_probs, entropy = self.actor.evaluate_actions(
                states, uav_ids, actions, action_mask=action_masks)
            new_values = self.critic(states, uav_ids).squeeze(-1)

            # 概率比 r_t(theta) = pi_new / pi_old
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO Clip 目标
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - PPO_EPS_CLIP, 1.0 + PPO_EPS_CLIP) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Critic 损失: MSE(V(s), R_hat)
            critic_loss = F.mse_loss(new_values, returns)

            # 熵正则化
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + PPO_CRITIC_COEF * critic_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), PPO_MAX_GRAD_NORM)
            nn.utils.clip_grad_norm_(self.critic.parameters(), PPO_MAX_GRAD_NORM)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy += entropy.mean().item()

        # 清空缓冲区
        self.buffer.clear()
        self.update_count += 1

        return {
            'policy_loss':  total_policy_loss / PPO_EPOCHS,
            'critic_loss':  total_critic_loss / PPO_EPOCHS,
            'entropy':      total_entropy / PPO_EPOCHS,
            'mean_advantage': advantages.mean().item(),
        }

    def get_action(self, state: np.ndarray, uav_id: int,
                   action_mask: np.ndarray = None) -> Tuple[int, float, float]:
        """根据当前策略选择动作

        Args:
            state:       状态向量 (state_dim,)
            uav_id:      无人机 ID
            action_mask: 可选动作掩码 (1+max_beams,) bool 数组
        Returns:
            action:   动作索引
            log_prob: 对数概率
            value:    状态价值
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        id_t = torch.LongTensor([uav_id]).to(self.device)

        mask_t = None
        if action_mask is not None:
            mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_t, log_prob_t = self.actor.get_action(state_t, id_t, mask_t)
            value_t = self.critic(state_t, id_t)

        return action_t.item(), log_prob_t.item(), value_t.item()

    def save(self, filepath: str):
        """保存模型参数"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'update_count': self.update_count,
        }, filepath)

    def load(self, filepath: str):
        """加载模型参数"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.update_count = checkpoint['update_count']

    def set_lr(self, lr: float):
        """动态调整学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
