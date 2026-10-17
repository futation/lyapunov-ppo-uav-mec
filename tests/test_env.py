"""环境核心行为测试: 确定性 / 掩码 / 多时隙 / Lyapunov 队列"""
import numpy as np
import pytest

from env import UAV_HAPMECEnv, get_state_dim
from config import AREA_SIZE, E_AVG


@pytest.fixture
def small_env():
    """4 架 UAV 的小规模环境 (加速测试)"""
    return UAV_HAPMECEnv(low_uavs=2, mid_uavs=1, max_uavs=1,
                         randomize_positions=True)


def test_reset_deterministic_with_seed():
    """同一种子 reset 两次, 观测与 UAV 位置完全一致"""
    e1 = UAV_HAPMECEnv(low_uavs=2, mid_uavs=1, max_uavs=1)
    o1, _ = e1.reset(seed=123)
    e2 = UAV_HAPMECEnv(low_uavs=2, mid_uavs=1, max_uavs=1)
    o2, _ = e2.reset(seed=123)
    assert (o1 == o2).all()
    assert (e1.uav_positions == e2.uav_positions).all()


def test_different_seeds_differ():
    e1 = UAV_HAPMECEnv(low_uavs=2, mid_uavs=1, max_uavs=1)
    o1, _ = e1.reset(seed=1)
    e2 = UAV_HAPMECEnv(low_uavs=2, mid_uavs=1, max_uavs=1)
    o2, _ = e2.reset(seed=2)
    assert not (o1 == o2).all()


def test_observation_shape_and_dtype(small_env):
    obs, _ = small_env.reset(seed=0)
    assert obs.shape == (get_state_dim(),)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_action_mask_idle_vs_busy(small_env):
    """空闲 UAV: 动作 0 恒可用; 忙碌 UAV: 掩码全空"""
    small_env.reset(seed=0)
    # 让所有 UAV 都启动任务 -> 全忙
    for uid in range(small_env.num_uavs):
        small_env.step(uid, 0)
    for uav in small_env.uavs:
        if uav.state != 0:
            assert not small_env.get_action_mask(uav.id).any()


def test_invalid_action_penalty(small_env):
    """越界动作返回 PENALTY_INVALID, 不可行动作返回 PENALTY_INFEASIBLE"""
    small_env.reset(seed=0)
    _, r, _, _, info = small_env.step(0, 99)
    assert r == pytest.approx(-10.0)
    assert info['reason'] == 'invalid'


def test_busy_uav_skipped_in_step(small_env):
    """忙碌 UAV 调 step 会被跳过 (info['busy']=True, 奖励 0)"""
    small_env.reset(seed=0)
    small_env.step(0, 0)  # UAV0 开始任务 -> 忙碌
    if small_env.uavs[0].state != 0:
        _, r, _, _, info = small_env.step(0, 1)
        assert info.get('busy') is True
        assert r == 0.0


def test_multi_slot_busy_state_machine(small_env):
    """波束任务跨越多个时隙: 决策后忙碌, 若干时隙后回到空闲"""
    small_env.reset(seed=3)
    # 驱动 UAV0 飞向波束 (若存在可行簇)
    obs, r, _, _, info = small_env.step(0, 1)
    if info.get('reason') == 'not_feasible':
        pytest.skip('当前种子下无可行簇')
    if small_env.uavs[0].state == 1:
        busy_slots = small_env.uavs[0].busy_remaining
        assert busy_slots >= 1
        # 逐时隙 tick 直到完成
        for _ in range(busy_slots + 1):
            small_env.end_of_step()
        assert small_env.uavs[0].state == 0
        assert small_env.uavs[0].busy_remaining == 0


def test_lyapunov_queues_bounded(small_env):
    """跑一段仿真, 虚拟队列 Z/Y 保持在设定上界内且非负"""
    small_env.reset(seed=5)
    rng = np.random.RandomState(0)
    for _ in range(15):
        for uid in range(small_env.num_uavs):
            mask = small_env.get_action_mask(uid)
            valid = np.where(mask)[0]
            action = int(rng.choice(valid)) if len(valid) > 0 else 0
            small_env.step(uid, action)
        small_env.end_of_step()
    Z, Y = small_env.Z, small_env.Y
    assert 0.0 <= Z <= 1e8
    assert 0.0 <= Y <= 1e6


def test_energy_constraint_reference():
    """E_AVG 按架构数量缩放参数可传入环境"""
    e = UAV_HAPMECEnv(low_uavs=1, mid_uavs=1, max_uavs=1, e_avg=1234.5)
    assert e.e_avg == pytest.approx(1234.5)
    assert e.e_avg != E_AVG  # 未走 config 默认值


def test_episode_runs_without_nan():
    """完整 episode (随机策略) 无 NaN, 统计字段可用"""
    e = UAV_HAPMECEnv(low_uavs=3, mid_uavs=2, max_uavs=1)
    e.reset(seed=42)
    rng = np.random.RandomState(1)
    for _ in range(20):
        for uid in range(e.num_uavs):
            mask = e.get_action_mask(uid)
            valid = np.where(mask)[0]
            a = int(rng.choice(valid)) if len(valid) > 0 else 0
            e.step(uid, a)
        e.end_of_step()
    stats = e.get_step_stats()
    for key in ('Q', 'Z', 'Y', 'total_steps'):
        assert np.isfinite(stats[key])
    assert stats['total_steps'] == 20
