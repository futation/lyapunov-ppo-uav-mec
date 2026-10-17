"""balanced_buffer.py — 按 UAV 均衡采样的经验缓冲区

核心机制 (异构多智能体 PPO 的稳定器):
  1. 每个 UAV 维护独立经验缓冲区 D_i;
  2. 每次更新从每个缓冲区均匀抽取 batch_size/N 条样本;
  3. 合并后随机打乱。

动机: 异构 UAV (算力差 10 倍) 的决策频率与回报分布差异很大,
若混合采样新, 高频/高回报 UAV 的经验会主导梯度, 低算力 UAV 的
决策模式学不到。均衡配额让每个"身份"对梯度的贡献相同, 降低方差。
"""

import numpy as np
import torch
from typing import List, Dict, Optional
from config import NUM_UAVS, PPO_BATCH_SIZE


class PerUAVBalancedBuffer:
    """按 UAV 均衡采样的经验缓冲区

    每个 UAV 维护独立的经验存储, 确保异构 UAV 的决策模式被公平学习。
    额外存储每个样本的动作掩码, 保证 PPO 更新时新旧策略对数概率
    在同一掩码分布下计算。
    """

    def __init__(self, num_uavs: int = NUM_UAVS,
                 batch_size: int = PPO_BATCH_SIZE,
                 num_actions: int = 5,
                 device: str = 'cpu'):
        """
        Args:
            num_uavs:    无人机数量 N
            batch_size:  总训练批次大小 B
            num_actions: 动作空间大小 (用于掩码存储)
            device:      存储设备
        """
        self.num_uavs = num_uavs
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.device = device

        # 每个 UAV 的独立缓冲区: 存储字典列表
        self.buffers: Dict[int, List[Dict]] = {i: [] for i in range(num_uavs)}

        # 每个 UAV 的样本数量计数
        self.counts = np.zeros(num_uavs, dtype=int)

    def add(self, uav_id: int,
            state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool,
            log_prob: float, value: float,
            action_mask: np.ndarray = None):
        """向指定 UAV 的缓冲区添加一条经验

        Args:
            uav_id:      无人机 ID (0-indexed)
            state:       状态向量 (state_dim,)
            action:      执行的动作 (整数)
            reward:      即时奖励
            next_state:  下一状态
            done:        是否终止
            log_prob:    动作对数概率 log pi(a|s)
            value:       Critic 估计的状态价值 V(s)
            action_mask: 采样时的动作掩码 (num_actions,), None 视为全可用
        """
        if uav_id < 0 or uav_id >= self.num_uavs:
            raise ValueError(f"Invalid uav_id: {uav_id}")

        if action_mask is None:
            action_mask = np.ones(self.num_actions, dtype=bool)

        self.buffers[uav_id].append({
            'state':       state.astype(np.float32),
            'action':      int(action),
            'reward':      float(reward),
            'next_state':  next_state.astype(np.float32),
            'done':        bool(done),
            'log_prob':    float(log_prob),
            'value':       float(value),
            'action_mask': np.asarray(action_mask, dtype=bool),
        })
        self.counts[uav_id] += 1

    def is_ready(self) -> bool:
        """检查是否所有 UAV 都有足够样本进行更新"""
        min_samples = max(1, self.batch_size // self.num_uavs)
        return bool(np.all(self.counts >= min_samples))

    def sample(self) -> Optional[Dict[str, torch.Tensor]]:
        """均衡采样: 从每个 UAV 缓冲区均匀抽取 quota 条 -> 合并 -> 打乱

        Returns:
            字典包含 'states', 'uav_ids', 'actions', 'rewards', 'next_states',
            'dones', 'old_log_probs', 'old_values', 'action_masks' — 全部为 Tensor;
            样本不足时返回 None
        """
        if not self.is_ready():
            return None

        per_uav = max(1, self.batch_size // self.num_uavs)
        batches = []

        for uav_id in range(self.num_uavs):
            buf = self.buffers[uav_id]
            n = min(per_uav, len(buf))
            if n == 0:
                continue
            indices = np.random.choice(len(buf), size=n, replace=False)
            for idx in indices:
                exp = buf[idx].copy()
                exp['uav_id'] = uav_id
                batches.append(exp)

        if len(batches) == 0:
            return None

        # 随机打乱, 避免按 UAV 分块的顺序偏差
        np.random.shuffle(batches)

        batch = {
            'states':        torch.FloatTensor(np.stack([e['state'] for e in batches])).to(self.device),
            'uav_ids':       torch.LongTensor([e['uav_id'] for e in batches]).to(self.device),
            'actions':       torch.LongTensor([e['action'] for e in batches]).to(self.device),
            'rewards':       torch.FloatTensor([e['reward'] for e in batches]).to(self.device),
            'next_states':   torch.FloatTensor(np.stack([e['next_state'] for e in batches])).to(self.device),
            'dones':         torch.BoolTensor([e['done'] for e in batches]).to(self.device),
            'old_log_probs': torch.FloatTensor([e['log_prob'] for e in batches]).to(self.device),
            'old_values':    torch.FloatTensor([e['value'] for e in batches]).to(self.device),
            'action_masks':  torch.BoolTensor(np.stack([e['action_mask'] for e in batches])).to(self.device),
        }

        return batch

    def clear(self):
        """清空所有缓冲区 (每次 PPO 更新后调用)"""
        for uav_id in range(self.num_uavs):
            self.buffers[uav_id].clear()
        self.counts.fill(0)

    def total_samples(self) -> int:
        """返回总样本数"""
        return int(np.sum(self.counts))

    def min_samples(self) -> int:
        """返回最少的 UAV 样本数"""
        return int(np.min(self.counts))
