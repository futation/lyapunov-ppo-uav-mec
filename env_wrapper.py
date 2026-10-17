"""env_wrapper.py — 标准 Gymnasium 单智能体视图

把原生多智能体环境 (step(uav_id, action) + end_of_step()) 包装成
标准 Gymnasium 接口, 便于接入 stable-baselines3 等现成 RL 库:

    import gymnasium as gym
    import env_wrapper  # 触发 gym.register
    env = gym.make("UavHapMec-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

时序约定 (与原生接口的对应关系):
  * 一次 step(action) = 当前空闲 UAV 的一次决策; 忙碌 UAV 自动跳过;
  * 同一时隙内所有 UAV 都决策完后, 自动调用 end_of_step() 推进时隙
    (某时隙开始时全部 UAV 忙碌则自动跳过该时隙);
  * 时隙奖励在时隙末才能结算, 无法在每架 UAV 决策时立即返回, 因此把
    整个时隙各 UAV 奖励之和计入该时隙最后一次决策的 step 返回值,
    逐 UAV 明细保留在 info["slot_rewards"]。对 PPO/DQN 这类折扣型
    算法这是可用的近似; 需要精确逐 UAV 奖励请用原生接口;
  * truncated 在达到 max_slots 个时隙后为 True, terminated 恒为 False。

观测 = 原生 15 维状态 + 1 维当前决策 UAV 的归一化 ID (使单个策略可以
区分不同算力档的执行者, 对应 networks.py 中 ID embedding 的设计)。
info["action_mask"] 提供每步的合法动作掩码 (MaskablePPO 惯例)。
"""

from typing import Dict, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    F_MIN_NUM, F_MID_NUM, F_MAX_NUM, LAMBDA_ARRIVAL,
    STEPS_PER_EPISODE, NUM_BEAMS,
)
from env import UAV_HAPMECEnv, get_state_dim


class UavHapMecGymEnv(gym.Env):
    """UAV-HAP MEC 卸载环境的标准 Gymnasium 单智能体封装"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, low_uavs: int = F_MIN_NUM, mid_uavs: int = F_MID_NUM,
                 max_uavs: int = F_MAX_NUM,
                 task_arrive_rate: float = LAMBDA_ARRIVAL, e_avg: float = None,
                 randomize_positions: bool = True,
                 max_slots: int = STEPS_PER_EPISODE,
                 render_mode: str = None):
        super().__init__()
        self.render_mode = render_mode
        self.max_slots = max_slots

        self.inner = UAV_HAPMECEnv(
            low_uavs=low_uavs, mid_uavs=mid_uavs, max_uavs=max_uavs,
            task_arrive_rate=task_arrive_rate, e_avg=e_avg,
            randomize_positions=randomize_positions,
            render_mode=render_mode,
        )

        self.num_uavs = self.inner.num_uavs
        self._state_dim = get_state_dim()
        self._cursor = 0        # 当前时隙内下一个待决策的 UAV 索引
        self._slot_total = 0.0  # 累计中的时隙奖励 (时隙末交付)
        self._slot_rewards = {}  # 上一时隙的逐 UAV 奖励明细
        self._slot_crossed = False  # 本 step 是否跨越了时隙边界

        # 观测 = 原生状态 + 当前 UAV 归一化 ID
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_dim + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(1 + NUM_BEAMS)

    # -------------------- 内部工具 --------------------
    def _append_actor_id(self, state: np.ndarray, uav_id: int) -> np.ndarray:
        actor_id = np.array([uav_id / max(1, self.num_uavs - 1)], dtype=np.float32)
        return np.concatenate([state, actor_id]).astype(np.float32)

    def _next_free_uav(self, start: int) -> int:
        """从 start 起找下一个空闲 UAV, 没有则返回 num_uavs"""
        i = start
        while i < self.num_uavs and self.inner.uavs[i].state != 0:
            i += 1
        return i

    def _current_obs_info(self) -> Tuple[np.ndarray, Dict]:
        uav_id = self._cursor
        state = self.inner._get_state_vector(uav_id)
        mask = self.inner.get_action_mask(uav_id)
        obs = self._append_actor_id(state, uav_id)
        info = {'uav_id': uav_id, 'action_mask': mask}
        return obs, info

    def _advance_to_decision(self) -> bool:
        """推进 cursor 到下一个可决策的 UAV。

        时隙内全部决策完 -> 调用 end_of_step() 结算并进入下一时隙
        (下一时隙开始时全忙则继续跳过)。达到 max_slots 后返回 False。
        """
        while True:
            self._cursor = self._next_free_uav(self._cursor)
            if self._cursor < self.num_uavs:
                return True  # 找到可决策的空闲 UAV

            # 本时隙决策完毕 -> 结算奖励
            rewards = self.inner.end_of_step()
            self._slot_total = float(sum(rewards.values()))
            self._slot_rewards = rewards
            self._slot_crossed = True

            if self.inner.total_steps >= self.max_slots:
                return False  # 达到时隙上限, episode 结束
            self._cursor = 0

    # -------------------- Gymnasium API --------------------
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.inner.reset(seed=seed)
        if options and 'max_slots' in options:
            self.max_slots = int(options['max_slots'])
        self._cursor = 0
        self._slot_total = 0.0
        self._slot_rewards = {}
        self._slot_crossed = False
        self._advance_to_decision()
        obs, info = self._current_obs_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.inner.step(self._cursor, int(action))

        self._slot_crossed = False
        self._cursor += 1
        has_decision = self._advance_to_decision()

        # 时隙奖励计入跨越时隙边界的这一次 step
        reward = self._slot_total if self._slot_crossed else 0.0
        slot_rewards = self._slot_rewards if self._slot_crossed else None

        if not has_decision:
            # 达到时隙上限: 返回终止过渡观测
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            mask = np.zeros(self.action_space.n, dtype=bool)
            info = {'uav_id': -1, 'action_mask': mask, 'slot_rewards': slot_rewards or {}}
            return obs, reward, False, True, info

        obs, info = self._current_obs_info()
        if slot_rewards is not None:
            info['slot_rewards'] = slot_rewards
        return obs, reward, False, False, info

    def render(self):
        """委托给内部环境的 rgb_array 渲染"""
        if self.render_mode != "rgb_array":
            raise NotImplementedError(
                f"仅支持 rgb_array 渲染, 当前 render_mode={self.render_mode!r}")
        return self.inner.render()


# -------------------- Gymnasium 注册 --------------------
try:
    gym.register(id='UavHapMec-v0', entry_point='env_wrapper:UavHapMecGymEnv')
except Exception:
    pass  # 重复导入时已注册
