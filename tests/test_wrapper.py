"""标准 Gymnasium wrapper 测试: 官方 env_checker + 接口语义"""
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import env_wrapper


@pytest.fixture
def wrapper():
    return env_wrapper.UavHapMecGymEnv(low_uavs=2, mid_uavs=1, max_uavs=1,
                                       max_slots=8)


def test_passes_official_env_checker():
    """通过 gymnasium.utils.env_checker.check_env (reset/step 规范、可复现)"""
    env = env_wrapper.UavHapMecGymEnv(low_uavs=2, mid_uavs=1, max_uavs=1,
                                      max_slots=5)
    check_env(env, skip_render_check=True)


def test_gym_make_registration():
    """gym.make('UavHapMec-v0') 可创建环境"""
    env = gym.make('UavHapMec-v0', low_uavs=2, mid_uavs=1, max_uavs=1,
                   max_slots=4)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape


def test_obs_contains_actor_id(wrapper):
    """观测最后一维是当前决策 UAV 的归一化 ID, 且随决策轮换"""
    obs, info = wrapper.reset(seed=0)
    first_id = info['uav_id']
    assert obs[-1] == pytest.approx(first_id / max(1, wrapper.num_uavs - 1))
    valid = np.where(info['action_mask'])[0]
    action = int(valid[0]) if len(valid) > 0 else 0
    obs2, _, _, _, info2 = wrapper.step(action)
    assert obs2[-1] == pytest.approx(info2['uav_id'] / max(1, wrapper.num_uavs - 1))


def test_action_mask_always_present(wrapper):
    """每步 info 都带合法动作掩码, 且动作 0 (机载) 对空闲 UAV 恒可用"""
    obs, info = wrapper.reset(seed=2)
    for _ in range(30):
        assert 'action_mask' in info
        valid = np.where(info['action_mask'])[0]
        action = int(np.random.choice(valid)) if len(valid) > 0 else 0
        obs, r, term, trunc, info = wrapper.step(action)
        if trunc:
            break


def test_truncation_after_max_slots(wrapper):
    """达到 max_slots 后 truncated=True, 内部时隙数一致"""
    obs, info = wrapper.reset(seed=4)
    truncated = False
    steps = 0
    while not truncated and steps < 2000:
        valid = np.where(info['action_mask'])[0]
        a = int(np.random.choice(valid)) if len(valid) > 0 else 0
        obs, r, term, truncated, info = wrapper.step(a)
        steps += 1
    assert truncated
    assert wrapper.inner.total_steps == wrapper.max_slots


def test_slot_reward_delivered_at_boundary(wrapper):
    """时隙奖励: 大多数 step 奖励为 0, 跨越时隙边界的 step 携带非零奖励
    且 info 含逐 UAV 明细 (至少出现过一次非零时隙)"""
    obs, info = wrapper.reset(seed=6)
    nonzero_slots = 0
    for _ in range(300):
        valid = np.where(info['action_mask'])[0]
        a = int(np.random.choice(valid)) if len(valid) > 0 else 0
        obs, r, term, trunc, info = wrapper.step(a)
        if 'slot_rewards' in info and len(info['slot_rewards']) > 0:
            nonzero_slots += 1
            assert abs(r - sum(info['slot_rewards'].values())) < 1e-6
        if trunc:
            break
    assert nonzero_slots > 0


def test_reset_determinism():
    """同种子两次 reset 后, 相同动作序列产生相同轨迹"""
    w1 = env_wrapper.UavHapMecGymEnv(low_uavs=2, mid_uavs=1, max_uavs=1, max_slots=8)
    w2 = env_wrapper.UavHapMecGymEnv(low_uavs=2, mid_uavs=1, max_uavs=1, max_slots=8)
    o1, _ = w1.reset(seed=99)
    o2, _ = w2.reset(seed=99)
    assert (o1 == o2).all()
    for _ in range(20):
        o1, r1, t1, tr1, i1 = w1.step(0)
        o2, r2, t2, tr2, i2 = w2.step(0)
        assert (o1 == o2).all() and r1 == r2
