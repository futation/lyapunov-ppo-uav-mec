"""channel_task.py — 信道模型与泊松任务生成

信道链路: 自由空间路损 -> 天线增益 -> SNR -> 香农容量。
任务模型: 每时隙按泊松过程到达, 位置支持"波束高斯热点 + 均匀散落"混合分布。
"""

from config import *
import math

def calculate_distance(x_1, x_2, y_1, y_2, h_1, h_2):
    D = math.sqrt((x_1 - x_2)**2 + (y_1 - y_2)**2 + (h_1 - h_2)**2)
    return D
# ==================== 信道模型 ====================
def free_space_path_loss_linear(distance_m, freq_hz):
    """自由空间路径损耗 (Friis), 返回线性功率损耗"""
    value = (4 * math.pi * distance_m * freq_hz) / C
    linear_loss = value ** 2   # 因为 value = 4πdf/c，平方后得到线性功率损耗
    return linear_loss

def channel_power_gain(G_tx, G_rx, path_loss_linear):
    """
    计算信道功率增益 |H|^2
    
    参数:
    G_tx: 发射天线增益（线性值）
    G_rx: 接收天线增益（线性值）
    L_f_linear: 路径损耗（线性值）
    
    返回:
    信道功率增益 |H|^2（线性值）
    """
    H_squared = (G_tx * G_rx) / path_loss_linear
    return H_squared

def calculate_snr(P_tx, H_squared, N0, B):
    """
    计算信噪比 SNR
    
    参数:
    P_tx: 发射功率（瓦特，线性值）
    H_squared: 信道功率增益 |H|^2（线性值）
    N0: 噪声功率谱密度（W/Hz）
    B: 带宽（Hz）
    
    返回:
    SNR（线性值）
    """
    noise_power = N0 * B  # 总噪声功率
    snr_linear = (P_tx * H_squared) / noise_power
    return snr_linear

def shannon_capacity(bandwidth_hz, snr_linear):
    """
    计算香农信道容量
    
    参数:
    bandwidth_hz: 带宽 (Hz)
    snr_linear: 信噪比（线性值，例如 SNR = 信号功率 / 噪声功率）
    
    返回:
    信道容量 (bps)
    """
    if snr_linear <= 0:
        return 0.0
    transmission_rate = bandwidth_hz * math.log2(1 + snr_linear)
    return transmission_rate  #单位：bps

# 如果需要从dB转换到线性值
def linear_to_db(linear_value):
    """将线性值转换为dB值"""
    if linear_value <= 0:
        return -float('inf')
    return 10 * math.log10(linear_value)

def db_to_linear(db_value):
    """将dB值转换为线性值"""
    return 10 ** (db_value / 10)

def R_transmission_rate_1(distance, G_tx, G_rx, P_tx, B_uplink):   #设备1可以为任务,上行链路，G_E或G_H
    L_f = free_space_path_loss_linear(distance, F_c)
    h   = channel_power_gain(G_tx, G_rx, L_f)
    snr = calculate_snr(P_tx, h, N0, B_uplink)
    R   = shannon_capacity(B_uplink, snr)
    return R #单位bps

def R_transmission_rate_2(distance, G_tx, G_rx, P_tx, B_downlink, UAV_num):   #设备1为HAP,下行链路，H_E链路,中继模式专用
    L_f = free_space_path_loss_linear(distance, F_c)
    h   = channel_power_gain(G_tx, G_rx, L_f)
    B_current = B_downlink / UAV_num
    snr = calculate_snr(P_tx, h, N0, B_current)
    R   = shannon_capacity(B_current, snr)
    return R #单位bps

######################任务############################
# 定义任务结构化数组的 dtype（全局一次定义）
TASK_DTYPE = np.dtype([
    ('id', 'i4'),                # 任务唯一ID
    ('x', 'f4'),                 # x坐标 (m)
    ('y', 'f4'),                 # y坐标 (m)
    ('data_bits', 'f8'),         # 数据量 (bits)
    ('cycles_per_bit', 'f4'),    # 每比特所需CPU周期
    ('arrival_time', 'f8'),      # 到达时间 (秒)
    ('total_gflops', 'f8')       # 总计算量 (GFLOPS)
])

def generate_tasks_at_step(lambda_rate, area_size=3000,
                          data_mb_range=(1, 10),
                          cycles_per_bit_range=(500, 2000),
                          current_time=0.0,
                          start_id=0,
                          beam_positions=None,
                          beam_task_ratio=0.0,
                          beam_sigma=40.0,
                          rng=None):
    """
    在当前时间步内，根据泊松分布生成新到达的任务。

    参数:
        lambda_rate: 平均到达率 (任务/秒)
        area_size: 正方形区域边长 (米)
        data_mb_range: 任务数据量范围 (MB)
        cycles_per_bit_range: 每比特CPU周期范围
        current_time: 当前时间（秒），写入任务的 arrival_time
        start_id: 分配给第一个新任务的ID起始值
        beam_positions: 波束中心列表 [(x,y),...], None=均匀分布
        beam_task_ratio: 波束区域任务占比 (0~1)
        beam_sigma: 波束周围高斯分布标准差 (m)
        rng: numpy 随机源 (np.random 或 Generator), None=全局 np.random。
             环境调用时应传入自己的 np_random 以保证 reset(seed) 完全可复现。

    返回:
        tasks: np.ndarray (结构化数组)
        next_id: 下一个可用的任务ID
    """
    rng = rng if rng is not None else np.random
    n = rng.poisson(lam=lambda_rate)
    if n == 0:
        return np.empty(0, dtype=TASK_DTYPE), start_id

    tasks = np.zeros(n, dtype=TASK_DTYPE)
    tasks['id'] = np.arange(start_id, start_id + n)

    # ---- 任务位置: 波束集中 或 均匀分布 ----
    if beam_positions and beam_task_ratio > 0:
        n_beam = int(n * beam_task_ratio)
        n_scattered = n - n_beam
        n_beams = len(beam_positions)

        # 波束任务: 以每个波束为中心的高斯分布
        for i in range(n_beam):
            bx, by = beam_positions[i % n_beams]  # 轮询分配到各波束
            tasks['x'][i] = np.clip(rng.normal(bx, beam_sigma), 0, area_size)
            tasks['y'][i] = np.clip(rng.normal(by, beam_sigma), 0, area_size)

        # 零散任务: 均匀分布
        for i in range(n_beam, n):
            tasks['x'][i] = rng.uniform(0, area_size)
            tasks['y'][i] = rng.uniform(0, area_size)
    else:
        # 传统均匀分布
        tasks['x'] = rng.uniform(0, area_size, n)
        tasks['y'] = rng.uniform(0, area_size, n)

    # 随机数据量 (MB -> bits: 1 MB = 8×10^6 bits)
    data_mb = rng.uniform(data_mb_range[0], data_mb_range[1], n)
    tasks['data_bits'] = data_mb * 8e6
    # 随机计算密度
    tasks['cycles_per_bit'] = rng.uniform(cycles_per_bit_range[0], cycles_per_bit_range[1], n)
    # 到达时间
    tasks['arrival_time'] = current_time
    # 计算 GFLOPS
    tasks['total_gflops'] = tasks['data_bits'] * tasks['cycles_per_bit'] / 1e9

    return tasks, start_id + n

if __name__ == "__main__":
    # ========== 使用示例 ==========
    # 设置随机种子以便复现
    np.random.seed(42)

    # 模拟参数
    avg_arrival_rate = 5.0      # 平均每秒到达5个任务
    total_simulation_seconds = 10
    current_time = 0.0
    next_task_id = 0

    all_tasks = []  # 收集所有到达的任务

    # 时间步进模拟（每秒一个步骤）
    for step in range(total_simulation_seconds):
        tasks_at_step, next_task_id = generate_tasks_at_step(
            lambda_rate=avg_arrival_rate,
            area_size=3000,
            data_mb_range=(0.5, 5),      # 自定义范围
            cycles_per_bit_range=(800, 2500),
            current_time=current_time,
            start_id=next_task_id
        )
        if len(tasks_at_step) > 0:
            all_tasks.append(tasks_at_step)
            print(f"时间 {current_time:.1f}s: 到达 {len(tasks_at_step)} 个任务, 下一个ID = {next_task_id}")
        current_time += 1.0

    # 合并所有任务为一个数组
    if all_tasks:
        all_tasks_arr = np.concatenate(all_tasks)
        print("\n任务摘要:")
        print(f"总任务数: {len(all_tasks_arr)}")
        print(f"ID范围: {all_tasks_arr['id'].min()} ~ {all_tasks_arr['id'].max()}")
        print(f"数据量范围: {all_tasks_arr['data_bits'].min()/1e6:.2f} ~ {all_tasks_arr['data_bits'].max()/1e6:.2f} MB")
        print(f"计算需求范围: {all_tasks_arr['total_gflops'].min():.2f} ~ {all_tasks_arr['total_gflops'].max():.2f} GFLOPS")
    else:
        print("没有任务到达。")
