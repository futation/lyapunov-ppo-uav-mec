"""PPO 训练器单元测试: 掩码一致性 / 更新可行性 / 模型存取"""
import numpy as np
import pytest
import torch

from ppo_trainer import PPOTrainer
from config import PPO_EPS_CLIP


@pytest.fixture
def trainer():
    torch.manual_seed(0)
    np.random.seed(0)
    t = PPOTrainer(state_dim=15, num_uavs=4)
    return t


def test_masked_action_never_outside_mask(trainer):
    """被掩码动作的采样概率为 0 (logits 置 -inf)"""
    mask = np.array([True, False, True, False, False])
    picked = set()
    for _ in range(50):
        a, lp, v = trainer.get_action(np.random.randn(15).astype(np.float32),
                                      uav_id=0, action_mask=mask)
        picked.add(a)
    assert picked <= {0, 2}


def test_update_runs_with_finite_losses(trainer):
    """填充缓冲区后执行一次 PPO 更新, 各损失项有限, 缓冲区清空"""
    mask = np.array([True, True, True, False, False])
    for _ in range(trainer.buffer.batch_size // trainer.buffer.num_uavs + 1):
        for uid in range(4):
            state = np.random.randn(15).astype(np.float32)
            a, lp, v = trainer.get_action(state, uid, mask)
            trainer.store_experience(uid, state, a, reward=float(np.random.randn()),
                                     next_state=state, done=False,
                                     log_prob=lp, value=v, action_mask=mask)
    assert trainer.is_update_ready()
    info = trainer.update()
    assert info is not None
    for key in ('policy_loss', 'critic_loss', 'entropy', 'mean_advantage'):
        assert np.isfinite(info[key])
    assert trainer.buffer.total_samples() == 0
    assert trainer.update_count == 1


def test_mask_consistency_between_sample_and_update(trainer):
    """更新时传入与采样相同的掩码: 对被掩码动作, 新旧对数概率都是 -inf,
    但有限动作上的 ratio 仍有效 (不产生 NaN)"""
    mask = np.array([True, False, True, False, False])
    state = np.random.randn(15).astype(np.float32)
    a, lp, v = trainer.get_action(state, 0, mask)
    trainer.store_experience(0, state, a, 0.5, state, False, lp, v,
                             action_mask=mask)
    # 凑齐各 UAV 样本 (默认 batch_size=128, 每架需 >= 32 条)
    full_mask = np.ones(5, dtype=bool)
    for uid in range(1, 4):
        for _ in range(33):
            s = np.random.randn(15).astype(np.float32)
            aa, lpp, vv = trainer.get_action(s, uid, full_mask)
            trainer.store_experience(uid, s, aa, 0.1, s, False, lpp, vv,
                                     action_mask=full_mask)
    for _ in range(32):
        s = np.random.randn(15).astype(np.float32)
        aa, lpp, vv = trainer.get_action(s, 0, mask)
        trainer.store_experience(0, s, aa, 0.1, s, False, lpp, vv,
                                 action_mask=mask)

    info = trainer.update()
    assert info is not None
    assert np.isfinite(info['policy_loss'])


def test_save_and_load_roundtrip(trainer, tmp_path):
    """save/load 后策略输出一致"""
    path = str(tmp_path / 'model.pt')
    trainer.save(path)

    torch.manual_seed(0)
    t2 = PPOTrainer(state_dim=15, num_uavs=4)
    t2.load(path)

    state = np.random.randn(15).astype(np.float32)
    torch.manual_seed(123)
    a1, lp1, v1 = trainer.get_action(state, 0)
    torch.manual_seed(123)
    a2, lp2, v2 = t2.get_action(state, 0)
    assert a1 == a2 and lp1 == pytest.approx(lp2) and v1 == pytest.approx(v2)


def test_lr_and_entropy_adjustment(trainer):
    """动态学习率与熵系数可调"""
    trainer.set_lr(1e-4)
    assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(1e-4)
    trainer.set_entropy_coef(0.02)
    assert trainer.entropy_coef == pytest.approx(0.02)
