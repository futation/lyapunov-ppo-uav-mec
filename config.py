"""config.py — 全局仿真参数

所有参数按物理含义分组。环境/算法模块通过 `from config import *` 引用，
调参只需改这一个文件。
"""

import numpy as np

# ==================== 物理常数 ====================
C       = 3e8      # 光速 (m/s)
G       = 9.8      # 重力加速度 (m/s^2)
RHO_AIR = 1.225    # 空气密度 (kg/m^3)

# ==================== Lyapunov 优化参数 ====================
# 虚拟队列 Z(t) 追踪长期平均能耗约束: 平均每时隙能耗不得超过 E_AVG
E_AVG = 30000.0    # 长期平均能耗上限 (J/slot), 按 10 架 UAV x ~3000J 量级设定

# ==================== 系统布局 ====================
AREA_SIZE    = 1000     # 覆盖区域边长 (m), 波束中心坐标以此坐标系为准
NUM_UAVS     = 10       # 无人机总数 (低+中+高算力三档, 见 F_*_NUM)
UAV_SPEED    = 10       # 无人机飞行速度 (m/s)
UAV_ALTITUDE = 300      # 无人机飞行高度 (m)
H_hap        = 15000    # HAP 平台高度 (m)

# ==================== 固定波束配置 ====================
# 4 个波束中心对称分布在区域四角附近, 任务以 BEAM_TASK_RATIO 的比例
# 集中生成在波束周围 (高斯热点, sigma=BEAM_SIGMA)
BEAM_POSITIONS  = [(200, 200), (800, 200), (800, 800), (200, 800)]  # 波束中心 (m)
BEAM_RADIUS     = 95      # 波束覆盖半径 (m)
NUM_BEAMS       = 4       # 固定波束数量
BEAM_TASK_RATIO = 0.85    # 集中在波束附近生成的任务占比
BEAM_SIGMA      = 40.0    # 波束热点高斯分布 sigma (m), ~95% 落在 2 sigma 内

# ==================== 任务参数 ====================
LAMBDA_ARRIVAL = 40       # 默认任务到达率 (GFLOPS/s)
W_BASE         = 45.0     # 基准任务包计算量 (GFLOPS)
W_MAX          = 63.0     # 最大簇任务量 (GFLOPS), 聚类时作为簇规模上限

# ==================== 通信参数 ====================
B     = 20e6    # HAP 系统带宽 (Hz)
B_UAV = 10e6    # 无人机间/本地链路带宽 (Hz)
N0    = 4e-21   # 噪声功率 (W), 约 -160 dBm
F_c   = 2.4e9   # 载波频率 (Hz)

# ==================== 无人机动力学与算力 ====================
R_Beam     = 95       # UAV 本地服务覆盖半径 (m), 与波束半径一致
M          = 2        # 无人机重量 (kg)
C_D        = 0.5      # 等效阻力系数
S          = 0.5      # 总转子盘面积 (m^2)
R_ROTOR    = 0.2      # 单个旋翼半径 (m)
NUM_ROTORS = 4        # 旋翼数量
A          = NUM_ROTORS * 3.14159 * R_ROTOR**2   # 总转子盘面积 (m^2)
v_h        = (M * G / (2 * RHO_AIR * A)) ** 0.5  # 悬停诱导速度 (m/s)
kappa      = 1e-27    # 电容系数 (芯片能耗模型)

# 异构算力三档: 1~10 GHz 共 10 倍差距, 促使"高算力飞远簇 / 低算力飞近簇"的分工
F_MIN_NUM = 5      # 低算力无人机数量
F_MID_NUM = 3      # 中算力无人机数量
F_MAX_NUM = 2      # 高算力无人机数量
F_MIN     = 1e9    # 低算力 (Hz)
F_MID     = 5e9    # 中算力 (Hz)
F_MAX     = 10e9   # 高算力 (Hz)

# ==================== 天线增益与功率 (线性值, 非 dB) ====================
G_tx_ground = 1.0    # 地面节点发射增益 (0 dBi)
G_rx_ground = 1.0    # 地面节点接收增益 (0 dBi)
P_tx_ground = 0.1    # 地面节点发射功率 (W)

G_tx_uav = 2.0       # UAV 发射增益 (~3 dBi)
G_rx_uav = 2.0       # UAV 接收增益 (~3 dBi)
P_tx_uav = 0.5       # UAV 发射功率 (W)

G_tx_hap = 10.0      # HAP 发射增益 (10 dBi)
G_rx_hap = 10.0      # HAP 接收增益 (10 dBi)
P_tx_hap = 10        # HAP 发射功率 (W)

L_f = 1.0            # 馈线损耗 (统一简化为无损耗)

# ==================== 时隙参数 ====================
SLOT_DURATION = 5.0  # 每个时隙持续时间 (s)
# 量级参考: 飞行 100m (10s) + 计算 20 GFLOPS / 5 GHz (4s) ≈ 14s ≈ 3 个时隙

# ==================== 奖励函数参数 ====================
# 每个完成任务的 UAV 获得形如下式的即时奖励:
#   r = OMEGA * task_vol
#       - P_D * (tanh(fly_dist / FLY_REF_DIST) - 1)      # 卸载距离惩罚 (越远越亏)
#       + P_L * exp(-(task_vol/REF_TASK_VOL)^3)          # 负载贡献奖励 (簇越满越高)
#       - LAMBDA_DELAY  * delay  / REF_DELAY             # 归一化时延惩罚
#       - LAMBDA_ENERGY * energy / REF_ENERGY            # 归一化能耗惩罚
# 时延/能耗用常量参考值归一化 (而非按 task_vol 缩放), 避免多 UAV 场景下
# task_vol 变小时惩罚项爆炸 (credit assignment 恶化)
REWARD_OMEGA        = 0.1     # 单位任务量奖励系数
REWARD_P_D          = 2.5     # 卸载距离惩罚幅度
REWARD_FLY_REF_DIST = 800.0   # 距离惩罚的 tanh 参考距离 (m)
REWARD_P_L          = 3.5     # 负载贡献奖励幅度
REF_TASK_VOL        = 100.0   # 负载奖励的参考任务量 (GFLOPS)
REF_DELAY           = 10.0    # 时延参考值 (s), 典型单 UAV 任务延迟
REF_ENERGY          = 5000.0  # 能耗参考值 (J), 典型单 UAV 任务能耗
LAMBDA_DELAY        = 0.3     # 时延惩罚系数
LAMBDA_ENERGY       = 1.0     # 能耗惩罚系数

# 非法/不可行动作与空任务的惩罚 (动作掩码之外的安全兜底)
PENALTY_INFEASIBLE = -5.0   # 掩码允许但不可行的簇组合
PENALTY_INVALID    = -10.0  # 越界动作
PENALTY_NO_TASK    = -1.0   # 决策时未接到任何任务

# ==================== PPO 超参数 ====================
PPO_LR               = 3e-4   # 初始学习率
PPO_LR_MIN           = 3e-5   # 学习率下限 (线性衰减终点)
PPO_EPS_CLIP         = 0.2    # PPO clip 范围 epsilon
PPO_GAMMA            = 0.99   # 折扣因子 gamma
PPO_LAMBDA_GAE       = 0.95   # GAE 优势估计参数 lambda
PPO_ENTROPY_COEF     = 0.05   # 熵正则系数 (线性退火起点)
PPO_ENTROPY_COEF_END = 0.01   # 熵正则系数 (退火终点, 保留少量探索)
PPO_CRITIC_COEF      = 0.5    # Critic 损失系数
PPO_BATCH_SIZE       = 128    # 每次更新的批次大小
PPO_EPOCHS           = 6      # 每次更新的 epoch 数
PPO_MAX_GRAD_NORM    = 0.5    # 梯度裁剪阈值

# ==================== 训练参数 ====================
NUM_EPISODES      = 500    # 训练 episode 数
STEPS_PER_EPISODE = 50     # 每个 episode 的时隙数
UPDATE_INTERVAL   = 256    # 经验累计多少步后触发一次 PPO 更新

# ==================== 动作空间 ====================
NUM_ACTIONS_BASE = 1   # 动作 0: 机载计算 (本地/中继由 HAP 自动裁决)
MAX_BEAMS        = 4   # 动作 1..NUM_BEAMS: 飞向对应波束
