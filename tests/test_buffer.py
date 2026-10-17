"""均衡采样缓冲区测试: 配额公平性 / 掩码存储 / 清空"""
import numpy as np
import pytest
import torch

from balanced_buffer import PerUAVBalancedBuffer


def _add_samples(buf: PerUAVBalancedBuffer, uav_id: int, n: int, state_dim=4):
    state = np.zeros(state_dim, dtype=np.float32)
    for i in range(n):
        buf.add(uav_id, state, action=i % 5, reward=0.1 * i,
                next_state=state, done=False,
                log_prob=-0.5, value=1.0,
                action_mask=np.array([True, True, False, False, False]))


def test_balanced_quota_sampling():
    """每个 UAV 贡献 batch_size/N 条样本 (均衡配额)"""
    n_uavs, batch = 4, 32
    buf = PerUAVBalancedBuffer(num_uavs=n_uavs, batch_size=batch, num_actions=5)
    for uid in range(n_uavs):
        _add_samples(buf, uid, 10)

    batch_data = buf.sample()
    assert batch_data is not None
    counts = torch.bincount(batch_data['uav_ids'], minlength=n_uavs)
    per_uav = max(1, batch // n_uavs)
    assert (counts == per_uav).all()  # 每架 UAV 恰好 quota 条


def test_not_ready_until_all_uavs_have_samples():
    """任何一架 UAV 样本不足时都不触发更新"""
    buf = PerUAVBalancedBuffer(num_uavs=3, batch_size=12, num_actions=5)
    _add_samples(buf, 0, 10)
    _add_samples(buf, 1, 10)
    assert not buf.is_ready()
    _add_samples(buf, 2, 10)
    assert buf.is_ready()


def test_action_mask_roundtrip():
    """掩码随经验存储并在 sample 输出中原样恢复"""
    buf = PerUAVBalancedBuffer(num_uavs=2, batch_size=2, num_actions=5)
    mask = np.array([True, False, True, False, False])
    buf.add(0, np.zeros(4, dtype=np.float32), 2, 1.0,
            np.zeros(4, dtype=np.float32), False, -0.1, 0.5, action_mask=mask)
    buf.add(1, np.zeros(4, dtype=np.float32), 1, 1.0,
            np.zeros(4, dtype=np.float32), False, -0.1, 0.5, action_mask=None)

    batch = buf.sample()
    assert batch is not None
    masks = batch['action_masks'].numpy()
    assert masks.shape == (2, 5)
    # 两条经验中必有一条是我们显式传入的掩码
    assert (masks == mask).any()
    # None 掩码默认全可用
    assert masks.all(axis=1).any()


def test_clear_resets_counts():
    buf = PerUAVBalancedBuffer(num_uavs=2, batch_size=8, num_actions=5)
    _add_samples(buf, 0, 6)
    _add_samples(buf, 1, 6)
    assert buf.total_samples() == 12
    buf.clear()
    assert buf.total_samples() == 0
    assert buf.min_samples() == 0
    assert buf.sample() is None


def test_invalid_uav_id_raises():
    buf = PerUAVBalancedBuffer(num_uavs=2, batch_size=8, num_actions=5)
    with pytest.raises(ValueError):
        buf.add(7, np.zeros(4, dtype=np.float32), 0, 0.0,
                np.zeros(4, dtype=np.float32), False, 0.0, 0.0)
