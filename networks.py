"""networks.py — Actor-Critic 网络 (PyTorch)

策略与价值网络均为 MLP, 通过 nn.Embedding 注入 UAV 身份:
所有 UAV 共享同一套网络参数 (参数共享), ID embedding 让网络能区分
不同算力档位的无人机 — 这是"异构多智能体单策略"的关键结构。

动作空间:
    0            = 机载计算 (本地或中继, 由 HAP 自动裁决)
    1 .. N_beams = 飞向对应波束/簇卸载
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

from config import NUM_UAVS, MAX_BEAMS


class ActorNetwork(nn.Module):
    """策略网络 (Actor): 输入状态 -> 输出动作概率分布 (支持动作掩码)"""

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 128,
                 num_uavs: int = NUM_UAVS,
                 max_beams: int = MAX_BEAMS,
                 embed_dim: int = 16):
        super().__init__()

        self.num_uavs = num_uavs
        self.max_beams = max_beams
        self.embed_dim = embed_dim

        # UAV ID Embedding (将离散 ID 映射为连续向量)
        self.id_embedding = nn.Embedding(num_uavs, embed_dim)

        # 状态编码: state_dim + embed_dim (ID嵌入) -> 隐藏层
        input_dim = state_dim + embed_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)

        # 输出头: 1 + max_beams 个动作 (0=机载计算, 1..max_beams=波束)
        self.num_actions = 1 + max_beams
        self.action_head = nn.Linear(hidden_dim // 2, self.num_actions)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, state: torch.Tensor, uav_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:    (batch, state_dim) 状态向量
            uav_ids:  (batch,) 无人机 ID (整数)
        Returns:
            logits:   (batch, 1+max_beams) 动作 logits
        """
        id_emb = self.id_embedding(uav_ids.long())  # (batch, embed_dim)
        x = torch.cat([state, id_emb], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        logits = self.action_head(x)
        return logits

    def get_action(self, state: torch.Tensor, uav_ids: torch.Tensor,
                   action_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作并返回对数概率

        Args:
            state:        (batch, state_dim)
            uav_ids:      (batch,)
            action_mask:  (batch, 1+max_beams) 可选动作掩码, True=可用
        Returns:
            actions:      (batch,) 动作索引
            log_probs:    (batch,) 对应动作的对数概率
        """
        logits = self.forward(state, uav_ids)

        # 应用动作掩码 (将不可用动作的 logits 设为 -inf, 采样概率为 0)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        return actions, log_probs

    def evaluate_actions(self, state: torch.Tensor, uav_ids: torch.Tensor,
                         actions: torch.Tensor, action_mask: torch.Tensor = None
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """评估给定动作的对数概率和熵 (用于 PPO 更新; 掩码须与采样时一致)

        Args:
            state:        (batch, state_dim)
            uav_ids:      (batch,)
            actions:      (batch,) 已执行的动作
            action_mask:  (batch, 1+max_beams) 与采样时相同的掩码
        Returns:
            log_probs:    (batch,) 动作对数概率
            entropy:      (batch,) 策略熵
        """
        logits = self.forward(state, uav_ids)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy


class CriticNetwork(nn.Module):
    """价值网络 (Critic): 输入状态 -> 状态价值 V(s), 同样带 ID embedding"""

    def __init__(self,
                 state_dim: int,
                 hidden_dim: int = 128,
                 num_uavs: int = NUM_UAVS,
                 embed_dim: int = 16):
        super().__init__()

        self.id_embedding = nn.Embedding(num_uavs, embed_dim)
        input_dim = state_dim + embed_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -0.1, 0.1)

    def forward(self, state: torch.Tensor, uav_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state:   (batch, state_dim)
            uav_ids: (batch,)
        Returns:
            values:  (batch, 1) 状态价值
        """
        id_emb = self.id_embedding(uav_ids.long())
        x = torch.cat([state, id_emb], dim=-1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        values = self.value_head(x)
        return values
