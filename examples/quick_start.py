"""quick_start.py — 最小可运行示例

演示原生多智能体接口的完整交互循环, 并对比三种朴素策略:
随机策略 / 贪心策略 (就近选择波束) / "全机载"策略。

运行:
    python examples/quick_start.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import NUM_BEAMS
from env import UAV_HAPMECEnv


def run_episode(env: UAV_HAPMECEnv, policy_fn, seed: int = 0, slots: int = 50) -> dict:
    """跑一个 episode (50 个时隙), 返回平均指标"""
    env.reset(seed=seed)
    rewards, delays, energies = [], [], []

    for _ in range(slots):
        # 每个时隙: 所有 UAV 依次决策 (忙碌 UAV 在 step 内自动跳过)
        for uav_id in range(env.num_uavs):
            mask = env.get_action_mask(uav_id)
            action = policy_fn(env, uav_id, mask)
            env.step(uav_id, action)

        # 所有 UAV 决策完 -> 推进时隙并结算奖励
        rewards_dict = env.end_of_step()
        rewards.extend(rewards_dict.values())
        delays.extend(env._step_completed_delays)
        energies.extend(env._step_completed_energies)

    return {
        'reward': float(np.mean(rewards)),
        'delay': float(np.mean(delays)) if delays else 0.0,
        'energy': float(np.mean(energies)) if energies else 0.0,
        'queue': float(env.Q),
    }


def random_policy(env, uav_id, mask):
    valid = np.where(mask)[0]
    return int(np.random.choice(valid)) if len(valid) > 0 else 0


def greedy_policy(env, uav_id, mask):
    """贪心: 飞向最近的可用波束; 没有可用波束则机载计算"""
    valid = np.where(mask)[0]
    beam_actions = [a for a in valid if 1 <= a <= NUM_BEAMS]
    if not beam_actions:
        return 0
    uav = env.uavs[uav_id]
    best = min(beam_actions,
               key=lambda a: np.hypot(uav.pos[0] - env.HAP.cluster_queue[a - 1]['center_x'],
                                      uav.pos[1] - env.HAP.cluster_queue[a - 1]['center_y']))
    return int(best)


def onboard_policy(env, uav_id, mask):
    """永远机载计算 (动作 0: 本地或中继由 HAP 自动裁决)"""
    return 0


if __name__ == '__main__':
    env = UAV_HAPMECEnv()  # 默认 10 架 UAV (5 低 + 3 中 + 2 高算力)

    for name, policy in [('random', random_policy),
                         ('greedy', greedy_policy),
                         ('onboard-only', onboard_policy)]:
        m = run_episode(env, policy, seed=42)
        print(f"[{name:>12}] reward={m['reward']:+8.2f}  "
              f"delay={m['delay']:7.1f}s  energy={m['energy']:8.0f}J  "
              f"residual_queue={m['queue']:7.0f} GFLOPS")
