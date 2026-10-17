"""信道模型与任务生成的物理合理性测试"""
import numpy as np

import pytest

from channel_task import (
    free_space_path_loss_linear, channel_power_gain, calculate_snr,
    shannon_capacity, R_transmission_rate_1, generate_tasks_at_step, TASK_DTYPE,
)
from config import F_c, N0


def test_path_loss_monotonic_in_distance():
    """距离越远, 自由空间路损越大"""
    losses = [free_space_path_loss_linear(d, F_c) for d in (100, 500, 1000, 5000)]
    assert losses == sorted(losses)


def test_channel_gain_and_snr_consistency():
    """增益 = G_tx*G_rx/路损; SNR 随功率单调增"""
    loss = free_space_path_loss_linear(1000.0, F_c)
    gain = channel_power_gain(G_tx=2.0, G_rx=2.0, path_loss_linear=loss)
    assert gain == pytest.approx(4.0 / loss)

    snrs = [calculate_snr(p, gain, N0, 1e6) for p in (0.1, 0.5, 1.0)]
    assert snrs == sorted(snrs)


def test_shannon_capacity_bounds():
    """容量非负、随 SNR 单调、SNR=0 时为 0"""
    assert shannon_capacity(1e6, 0.0) == 0.0
    caps = [shannon_capacity(1e6, snr) for snr in (0.1, 1.0, 10.0, 100.0)]
    assert caps == sorted(caps)
    assert all(c > 0 for c in caps)


def test_transmission_rate_positive_and_finite():
    """端到端速率有限且为正"""
    rate = R_transmission_rate_1(distance=500.0, G_tx=1.0, G_rx=2.0,
                                 P_tx=0.1, B_uplink=1e6)
    assert 0.0 < rate < 1e6 * np.log2(1 + 1e12)  # 带宽*log2(1+SNR) 上界内


def test_task_generation_poisson_and_hotspot():
    """任务数量服从泊松 (均值附近), 且大部分落在波束附近 (热点)"""
    np.random.seed(0)
    beams = [(200, 200), (800, 200), (800, 800), (200, 800)]

    counts = [len(generate_tasks_at_step(lambda_rate=20, area_size=1000,
                                         beam_positions=beams,
                                         beam_task_ratio=0.85, beam_sigma=40.0)[0])
              for _ in range(60)]
    assert 10 <= np.mean(counts) <= 30  # 均值 20 附近

    tasks, _ = generate_tasks_at_step(lambda_rate=100, area_size=1000,
                                      beam_positions=beams,
                                      beam_task_ratio=0.85, beam_sigma=40.0)
    assert len(tasks) > 0
    assert tasks['id'].min() >= 0
    # 热点比例: 距最近波束中心 <= 3*sigma 的任务应明显过半
    dists = np.min([np.hypot(tasks['x'] - bx, tasks['y'] - by)
                    for bx, by in beams], axis=0)
    assert (dists <= 120).mean() > 0.6
