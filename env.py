"""env.py — UAV-HAP 边缘计算卸载仿真环境 (原生多智能体接口)

场景: 一个 HAP (高空平台) 悬停在任务区域上空, 通过 4 个固定波束为
地面设备服务; 多架异构算力的 UAV 在低空游弋, 每个时隙自主决策:
    0          = 机载计算 (覆盖内有零散任务则本地算, 否则打包走 HAP 中继)
    1..N_beams = 飞向对应波束簇卸载 (飞行+传输+计算可能跨多个时隙)

集成 Lyapunov 虚拟队列优化:
    Z(t): 虚拟能耗队列, 追踪长期平均能耗不超过 E_AVG 的约束
    Y(t): 虚拟处理率队列, 追踪任务到达与处理速率的平衡
    两者进入观测向量, 使策略能感知长期约束的"债务水平"。

奖励 (每个完成任务的 UAV, 时隙末结算):
    r = OMEGA * task_vol
        - P_D * (tanh(fly_dist / FLY_REF_DIST) - 1)      # 卸载距离惩罚
        + P_L * exp(-(task_vol/REF_TASK_VOL)^3)          # 负载贡献奖励
        - LAMBDA_DELAY  * delay  / REF_DELAY             # 归一化时延惩罚
        - LAMBDA_ENERGY * energy / REF_ENERGY            # 归一化能耗惩罚

观测 (state_dim 维, 全部归一化到 ~[0,1]):
    [uav_load, x, y, beam_loads x N, beam_dists x N, Q, Z, Y, busy]

注意: 本类提供的是"多智能体原生接口" — step(uav_id, action) 每次让
一架 UAV 决策, 所有 UAV 决策完后调用 end_of_step() 推进一个时隙并结算
奖励 (返回 {uav_id: reward} 字典)。若需要标准 Gymnasium 单智能体接口
(step(action) 五元组), 请使用 env_wrapper.UavHapMecGymEnv。
"""

import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import matplotlib.figure
import matplotlib.backends.backend_agg
import matplotlib.patches

from config import *
from channel_task import *
from uav import UAV
from hap import HAP, CLUSTER_DTYPE


def get_state_dim() -> int:
    """返回状态向量维度: uav_load(1) + pos(2) + beam_loads(N) + beam_dists(N) + Q,Z,Y(3) + busy(1)"""
    return 1 + 2 + MAX_BEAMS + MAX_BEAMS + 3 + 1


def build_state_vector(uav_load: float, uav_x: float, uav_y: float,
                       beam_loads: np.ndarray, beam_dists: np.ndarray,
                       Q_t: float, Z_t: float, Y_t: float,
                       is_busy: bool = False) -> np.ndarray:
    """构建单个 UAV 的观测向量 (全部归一化到 ~[0,1])

    Args:
        uav_load:   当前 UAV 待处理负载 (GFLOPS)
        uav_x, y:   UAV 当前坐标 (m)
        beam_loads: 各波束/簇的负载列表
        beam_dists: UAV 到各波束中心的距离列表 (m)
        Q_t, Z_t, Y_t: Lyapunov 队列状态 (任务队列 / 能耗虚拟队列 / 处理率虚拟队列)
        is_busy:    UAV 是否忙碌 (多时隙模型)
    Returns:
        state: (state_dim,) float32 数组
    """
    # 波束负载补零/截断 + 归一化
    beam_load_arr = np.zeros(MAX_BEAMS, dtype=np.float32)
    if beam_loads is not None and len(beam_loads) > 0:
        n = min(len(beam_loads), MAX_BEAMS)
        beam_load_arr[:n] = np.array(beam_loads[:n], dtype=np.float32) / 100.0

    # 波束距离补零/截断 + 归一化 (用区域边长; 无簇时保持 1 表示"远")
    beam_dist_arr = np.ones(MAX_BEAMS, dtype=np.float32)
    if beam_dists is not None and len(beam_dists) > 0:
        n = min(len(beam_dists), MAX_BEAMS)
        raw = np.array(beam_dists[:n], dtype=np.float32)
        beam_dist_arr[:n] = np.clip(raw / AREA_SIZE, 0.0, 1.0)

    state = np.concatenate([
        [float(uav_load) / 500.0],         # UAV 负载
        [uav_x / AREA_SIZE],               # x 坐标
        [uav_y / AREA_SIZE],               # y 坐标
        beam_load_arr,                     # 波束负载 (N 维)
        beam_dist_arr,                     # 波束距离 (N 维)
        [float(Q_t) / 10000.0],            # 任务队列
        [float(Z_t) / 1e7],                # 能耗虚拟队列
        [float(Y_t) / 2000.0],             # 处理率虚拟队列
        [1.0 if is_busy else 0.0],         # 忙碌标志 (多时隙模型)
    ]).astype(np.float32)

    return state


class UAV_HAPMECEnv(gym.Env):
    """UAV-HAP 边缘计算卸载环境 (多智能体原生接口 + Lyapunov 虚拟队列)"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, low_uavs=F_MIN_NUM, mid_uavs=F_MID_NUM, max_uavs=F_MAX_NUM,
                 task_arrive_rate=LAMBDA_ARRIVAL, e_avg=None,
                 randomize_positions: bool = True, render_mode: str = None):
        """
        Args:
            low_uavs / mid_uavs / max_uavs: 三档算力的 UAV 数量 (合计为总数)
            task_arrive_rate: 任务到达率 (GFLOPS/s)
            e_avg: 长期平均能耗上限 (J/slot), None 则用 config.E_AVG
            randomize_positions: 每次 reset 是否重新随机 UAV 位置
                (False 时固定使用 __init__ 时生成的位置, 便于对照实验)
            render_mode: None 或 "rgb_array"
        """
        super().__init__()

        self.randomize_positions = randomize_positions
        self.render_mode = render_mode

        # Gymnasium 标准随机源 (reset(seed=...) 时由 super().reset 重新播种)
        self.np_random, _ = gym.utils.seeding.np_random(None)

        self.num_uavs = low_uavs + mid_uavs + max_uavs
        self.low_uavs, self.mid_uavs, self.max_uavs = low_uavs, mid_uavs, max_uavs
        self.task_arrive_rate = task_arrive_rate  # GFLOPS/s
        self.e_avg = e_avg if e_avg is not None else E_AVG  # 能耗上限, 可按 UAV 数量缩放

        # ---- HAP 初始化 (位于区域中心上方 H_hap 米) ----
        self.HAP = HAP(AREA_SIZE / 2, AREA_SIZE / 2)

        # ---- UAV 初始化 ----
        self.uavs: List[UAV] = []
        self.uav_positions = np.zeros((self.num_uavs, 2))
        for idx in range(self.num_uavs):
            uav = UAV(idx)
            uav.pos = [
                self.np_random.uniform(0, AREA_SIZE),
                self.np_random.uniform(0, AREA_SIZE),
                UAV_ALTITUDE
            ]
            uav.high = UAV_ALTITUDE

            # 三档异构算力: 低/中/高
            if idx < low_uavs:
                uav.Power_c = F_MIN
            elif idx < low_uavs + mid_uavs:
                uav.Power_c = F_MID
            else:
                uav.Power_c = F_MAX

            uav.p_fly = uav.P_fly()
            uav.initial_pos = list(uav.pos)
            self.uav_positions[idx] = [uav.pos[0], uav.pos[1]]
            self.uavs.append(uav)

        # ---- Lyapunov 虚拟队列 ----
        self.Q = 0.0   # 任务队列长度 (GFLOPS)
        self.Z = 0.0   # 虚拟能耗队列: Z(t+1) = max(0, Z(t) + E_total(t) - E_AVG)
        self.Y = 0.0   # 虚拟处理率队列: Y(t+1) = max(0, Y(t) + arrival - service)

        # ---- 动作空间: 0=机载计算, 1..NUM_BEAMS=飞向波束 n ----
        self.num_actions = 1 + NUM_BEAMS
        self.action_space = spaces.Discrete(self.num_actions)

        # ---- 观测空间 ----
        state_dim = get_state_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )

        # ---- 时隙内临时状态 ----
        self._step_uav_decisions = {}     # {uav_id: action}
        self._step_uav_delays = {}        # {uav_id: delay}
        self._step_uav_energies = {}      # {uav_id: energy}
        self._step_uav_task_volumes = {}  # {uav_id: gflops}

        # ---- 本时隙完成任务的延迟/能耗 (供训练脚本收集统计) ----
        self._step_completed_delays = []
        self._step_completed_energies = []

        # ---- 任务生成器状态 ----
        self._next_task_id = 0
        self._current_time = 0.0

        # ---- 统计 ----
        self.total_steps = 0
        self.episode_rewards = []         # 记录每个时隙的总奖励

        self.reset()

    # ==================== 重置 ====================
    def reset(self, seed=None, options=None):
        """重置环境。seed 同时驱动 Gymnasium 随机源与全局 numpy 随机源
        (任务生成模块当前使用全局 np.random, 为保证可复现一并播种)。"""
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._current_time = 0.0
        self._next_task_id = 0
        self.total_steps = 0
        self.episode_rewards = []

        # 重置 UAV (可选: 每 episode 重新随机位置, 增加场景多样性)
        for idx, uav in enumerate(self.uavs):
            if self.randomize_positions:
                uav.pos = [
                    self.np_random.uniform(0, AREA_SIZE),
                    self.np_random.uniform(0, AREA_SIZE),
                    UAV_ALTITUDE
                ]
                uav.initial_pos = list(uav.pos)
            else:
                uav.pos = list(uav.initial_pos)
            self.uav_positions[idx] = [uav.pos[0], uav.pos[1]]
            uav.state = 0
            uav.decision = 0
            uav.task = 0.0
            uav.cluster = []
            uav.cluster_num = 0
            uav.transmission_rate = 0.0
            uav.path_loss = 0.0
            uav.gain = 0.0
            uav.SNR = 0.0
            uav.delay = 0.0
            uav.energy = 0.0
            uav.utility = 0.0
            uav.need_time = 0.0
            uav.need_energy = 0.0
            uav.task_data_volume = 0.0
            # 多时隙忙碌追踪
            uav.busy_remaining = 0
            uav.is_flying = False
            uav._pending_task_vol = 0.0
            uav._pending_delay = 0.0
            uav._pending_energy = 0.0
            uav._action_taken = 0
            uav._fly_dist = 0.0

        # 重置 HAP
        self.HAP.relay_UAV_num = 0
        self.HAP.connected_drones = []
        self.HAP.task_queue = []
        self.HAP.cluster_queue = np.empty(0, dtype=CLUSTER_DTYPE)
        self.HAP.clustered_tasks_array = np.empty(0, dtype=TASK_DTYPE)
        self.HAP.remaining_task_queue = []
        self.HAP.taskpackage_queue = np.empty(0, dtype=TASK_DTYPE)

        # 重置 Lyapunov 队列
        self.Q = 0.0
        self.Z = 0.0
        self.Y = 0.0

        # 清空时隙临时状态
        self._step_uav_decisions.clear()
        self._step_uav_delays.clear()
        self._step_uav_energies.clear()
        self._step_uav_task_volumes.clear()
        self._step_completed_delays = []
        self._step_completed_energies = []

        # 生成初始任务
        self._generate_and_cluster_tasks()

        return self._get_state_vector(0), {}

    # ==================== 任务生成与聚类 ====================
    def _generate_and_cluster_tasks(self):
        """每个时隙开始时生成新任务并按固定波束聚类"""
        # 经验换算: GFLOPS/s 到达率 -> 每时隙任务个数 (缩放系数为调参常数)
        avg_tasks = max(1, int(self.task_arrive_rate / 4.0))
        lambda_tasks = max(0.5, float(avg_tasks))

        new_tasks, self._next_task_id = generate_tasks_at_step(
            lambda_rate=lambda_tasks,
            area_size=AREA_SIZE,
            data_mb_range=(1, 10),
            cycles_per_bit_range=(500, 2000),
            current_time=self._current_time,
            start_id=self._next_task_id,
            beam_positions=BEAM_POSITIONS,
            beam_task_ratio=BEAM_TASK_RATIO,
            beam_sigma=BEAM_SIGMA,
            rng=self.np_random,  # 使用环境自有随机源, reset(seed) 完全可复现
        )

        self.HAP.add_tasks_to_hap_queue(new_tasks)
        self.HAP.HAP_clustering()

        # 更新 Q(t)
        self.Q = sum(task[6] for task in self.HAP.task_queue)
        self._current_time += 1.0

    def _get_queue_state(self) -> Tuple[float, float, float]:
        return self.Q, self.Z, self.Y

    def _update_lyapunov_queues(self, total_energy: float, effective_rate: float):
        """更新虚拟队列 (加上限防溢出)。

        Z: 能耗债务 — 本时隙能耗超出 E_AVG 的部分累积;
        Y: 处理率债务 — 到达率超出实际处理速率的部分累积。
        """
        self.Z = np.clip(self.Z + total_energy - self.e_avg, 0.0, 1e8)
        self.Y = np.clip(self.Y + self.task_arrive_rate - effective_rate, 0.0, 1e6)

    # ==================== 观测与掩码 ====================
    def _get_state_vector(self, uav_id: int) -> np.ndarray:
        uav = self.uavs[uav_id]
        uav_load = uav._pending_task_vol if uav._pending_task_vol > 0 else 0.0
        uav_x, uav_y = uav.pos[0], uav.pos[1]

        beam_loads = []
        beam_dists = []
        if len(self.HAP.cluster_queue) > 0:
            for cluster in self.HAP.cluster_queue:
                beam_loads.append(float(cluster['total_gflops']))
                dist = np.hypot(uav_x - cluster['center_x'], uav_y - cluster['center_y'])
                beam_dists.append(dist)

        Q_t, Z_t, Y_t = self._get_queue_state()

        return build_state_vector(uav_load, uav_x, uav_y,
                                  np.array(beam_loads), np.array(beam_dists),
                                  Q_t, Z_t, Y_t,
                                  is_busy=(uav.state != 0))

    def get_action_mask(self, uav_id: int) -> np.ndarray:
        """生成动作掩码: 忙碌 UAV 全部不可用; 空闲 UAV 动作 0 恒可用,
        波束动作仅当该 UAV 算力允许的可行簇组合包含对应簇时可用"""
        mask = np.zeros(self.num_actions, dtype=bool)

        uav = self.uavs[uav_id]
        if uav.state != 0:
            return mask  # 忙碌 UAV: 所有动作不可用

        mask[0] = True  # 机载计算始终可用

        n_clusters = len(self.HAP.cluster_queue)
        if n_clusters > 0:
            feasible = uav.get_feasible_cluster_combinations(self.HAP.cluster_queue)
            for combo in feasible:
                for idx in combo['cluster_indices']:
                    if idx < NUM_BEAMS:
                        mask[1 + idx] = True

        return mask

    # ==================== 环境步进 (多时隙模型) ====================
    def step(self, uav_id: int, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """单个 UAV 执行动作。

        - 忙碌 UAV: 跳过, 返回当前状态 (info['busy']=True)
        - 空闲 UAV: 记录决策并启动任务 (本地/中继/飞向波束)
        实际奖励在所有 UAV 决策完后由 end_of_step() 统一结算。

        Returns:
            (observation, 0.0, False, False, info)
        """
        uav = self.uavs[uav_id]

        # ---- 忙碌 UAV 不参与决策 ----
        if uav.state != 0:
            return self._get_state_vector(uav_id), 0.0, False, False, {'busy': True}

        # ---- 空闲 UAV: 执行动作 ----
        n_clusters = len(self.HAP.cluster_queue)

        if action == 0:
            # 机载计算: 覆盖内有零散任务则本地计算, 否则打包走 HAP 中继
            package, remaining = self.HAP.collect_task_package_in_coverage(
                uav.pos[0], uav.pos[1], coverage_radius=R_Beam
            )
            if len(package) > 0:
                # === 本地计算 ===
                self.HAP.remaining_task_queue = remaining
                delay = sum(uav.local_transmission(t) for t in package)
                task_vol = sum(t['total_gflops'] for t in package)
                energy = uav.calculate_power(task_vol * 1e9)
                handled = set(t['id'] for t in package)
                self.HAP._remove_tasks_by_ids(list(handled))

                uav.start_onboard_task(delay, energy, task_vol, decision=1)
                self._step_uav_decisions[uav_id] = 0
                self._step_uav_delays[uav_id] = delay
                self._step_uav_energies[uav_id] = energy
                self._step_uav_task_volumes[uav_id] = task_vol
            else:
                # === 中继模式: 从零散队列打包任务, 双跳经 HAP 转发 ===
                max_pkg = int(uav.Power_c / F_MIN)
                target = W_BASE * max_pkg
                max_g = target + 15.0
                task_arr, remaining = self.HAP.task_package_by_size(target, max_g)
                if len(task_arr) > 0:
                    self.HAP.taskpackage_queue = np.concatenate(
                        (self.HAP.taskpackage_queue, task_arr))
                self.HAP.remaining_task_queue = remaining
                self.HAP.relay_UAV_num += 1
                self.HAP.connected_drones.append(uav_id)

                # 预估中继延迟 (先标记, end_of_step 中精确结算)
                task_vol = sum(t[6] for t in task_arr) if len(task_arr) > 0 else 0.0
                est_delay = sum(uav.local_transmission(t) for t in task_arr) if len(task_arr) > 0 else 1.0
                est_energy = uav.calculate_power(task_vol * 1e9) if task_vol > 0 else 0.0

                uav.start_onboard_task(est_delay, est_energy, task_vol, decision=3)
                self._step_uav_decisions[uav_id] = 0
                self._step_uav_delays[uav_id] = est_delay
                self._step_uav_energies[uav_id] = est_energy
                self._step_uav_task_volumes[uav_id] = task_vol

        elif 1 <= action <= n_clusters:
            # 飞向波束 n 卸载
            cluster_idx = action - 1
            feasible = uav.get_feasible_cluster_combinations(self.HAP.cluster_queue)
            selected = None
            for combo in feasible:
                if cluster_idx in combo['cluster_indices']:
                    selected = combo
                    break
            if selected is None:
                return self._get_state_vector(uav_id), PENALTY_INFEASIBLE, False, False, {'reason': 'not_feasible'}

            actual = [self.HAP.cluster_queue[i] for i in selected['cluster_indices']]
            tasks_arr = (np.array(self.HAP.task_queue, dtype=TASK_DTYPE)
                         if self.HAP.task_queue else np.empty(0, dtype=TASK_DTYPE))
            if len(actual) == 1:
                total_time, fly_time = uav.Select_single_cluster(actual[0], tasks_arr)
            else:
                total_time, fly_time = uav.Select_multi_cluster(actual, tasks_arr)

            task_vol = selected['total_gflops']
            fly_e = uav.p_fly * fly_time
            comp_e = uav.calculate_power(task_vol * 1e9)
            fly_dist = fly_time * UAV_SPEED

            # 开始波束任务 (多时隙忙碌)
            uav.start_beam_task(
                cluster_center_x=actual[0]['center_x'],
                cluster_center_y=actual[0]['center_y'],
                total_time=total_time, flight_time=fly_time,
                task_vol=task_vol, comp_energy=comp_e, fly_energy=fly_e,
                fly_dist=fly_dist, action=action
            )

            self.HAP._remove_clusters_by_indices(selected['cluster_indices'])
            self.HAP._remove_tasks_by_ids(selected['task_ids'])

            self._step_uav_decisions[uav_id] = action
            self._step_uav_delays[uav_id] = total_time
            self._step_uav_energies[uav_id] = comp_e + fly_e
            self._step_uav_task_volumes[uav_id] = task_vol

        else:
            return self._get_state_vector(uav_id), PENALTY_INVALID, False, False, {'reason': 'invalid'}

        return self._get_state_vector(uav_id), 0.0, False, False, {}

    # ==================== 时隙结束: 推进 + 结算 ====================
    def end_of_step(self) -> Dict[int, float]:
        """推进一个时隙并结算奖励:

        1. tick 所有 UAV (推进忙碌进度, 完成任务者转为空闲)
        2. 结算中继模式 (把中继 UAV 的预估值替换为精确的双跳时延/能耗)
        3. 对本时隙每个决策过的 UAV 计算奖励
        4. 更新 Lyapunov 虚拟队列 Z/Y
        5. 生成下一时隙任务

        Returns:
            {uav_id: reward} 字典
        """
        rewards = {}

        # ---- 1. Tick 所有 UAV ----
        completed_uavs = []  # 本时隙完成任务的 UAV
        for uav in self.uavs:
            was_busy = uav.busy_remaining > 0
            finished = uav.tick()
            if finished and was_busy:
                completed_uavs.append(uav.id)

        # ---- 2. 结算中继模式 (更新中继 UAV 的精确 delay/energy) ----
        if self.HAP.relay_UAV_num > 0 and len(self.HAP.taskpackage_queue) > 0:
            uavs_dict = {u.id: u for u in self.uavs}
            if len(self.HAP.task_queue) > 0:
                task_array = np.array(self.HAP.task_queue, dtype=TASK_DTYPE)
            else:
                task_array = self.HAP.taskpackage_queue
            relay_results = self.HAP.resolve_relay_mode(uavs_dict, task_array)
            for uid, res in relay_results.items():
                uav = self.uavs[uid]
                uav._pending_delay = res['delay']
                uav._pending_energy = res['energy']
                uav._pending_task_vol = res['task_data_volume']
                # 更新忙碌时隙数 (原来用的是估算值)
                uav._compute_slots = max(1, int(np.ceil(res['delay'] / SLOT_DURATION)))
                uav.busy_remaining = uav._compute_slots
                self._step_uav_delays[uid] = res['delay']
                self._step_uav_energies[uid] = res['energy']
                self._step_uav_task_volumes[uid] = res['task_data_volume']

        # ---- 3. 奖励计算: 对本时隙每个决策过的 UAV ----
        total_delay = 0.0
        total_energy = 0.0
        total_task = 0.0

        for uid in self._step_uav_decisions:
            uav = self.uavs[uid]
            # 使用 pending 数据 (包含中继结算后的精确值)
            task_vol = uav._pending_task_vol if uav._pending_task_vol > 0 else self._step_uav_task_volumes.get(uid, 0.0)
            delay_uav = uav._pending_delay if uav._pending_delay > 0 else self._step_uav_delays.get(uid, 0.0)
            energy_uav = uav._pending_energy if uav._pending_energy > 0 else self._step_uav_energies.get(uid, 0.0)
            action = self._step_uav_decisions[uid]

            total_delay += delay_uav
            total_energy += energy_uav
            total_task += task_vol

            if task_vol > 0:
                r = REWARD_OMEGA * task_vol

                if action >= 1:
                    # 卸载动作: 距离越远惩罚越大 (鼓励就近服务)
                    fly_dist = getattr(uav, '_fly_dist', 100.0)
                    D_np = REWARD_P_D * (math.tanh(fly_dist / REWARD_FLY_REF_DIST) - 1.0)
                    r -= D_np
                    # 负载贡献奖励: 承接的簇越满奖励越高
                    rate = task_vol / max(1.0, REF_TASK_VOL)
                    if rate <= 1.0:
                        P_np = REWARD_P_L * math.exp(-rate**3)
                    else:
                        P_np = REWARD_P_L * (1.0 - rate)
                    r += P_np

                # 归一化时延/能耗惩罚 (常量参考值, 不随 task_vol 缩放)
                r -= LAMBDA_DELAY * delay_uav / REF_DELAY
                r -= LAMBDA_ENERGY * energy_uav / REF_ENERGY

                rewards[uid] = r
            else:
                rewards[uid] = PENALTY_NO_TASK

        if not rewards:
            for uid in range(self.num_uavs):
                rewards[uid] = 0.0

        # ---- 3.5. 收集本时隙完成任务的延迟/能耗, 并清零 pending 数据 ----
        self._step_completed_delays = []
        self._step_completed_energies = []
        for uid in completed_uavs:
            uav = self.uavs[uid]
            if uav._pending_delay > 0:
                vol, delay, energy, action_taken, fly_dist = uav.collect_pending_reward()
                self._step_completed_delays.append(delay)
                self._step_completed_energies.append(energy)

        # ---- 4. Lyapunov 虚拟队列 ----
        effective_rate = total_task / max(0.1, total_delay / max(1, len(self._step_uav_decisions)))
        self._update_lyapunov_queues(total_energy, effective_rate)

        # ---- 5. 记录 ----
        total_ep_reward = sum(rewards.values())
        self.episode_rewards.append(total_ep_reward)

        # ---- 6. 清理 & 生成下一时隙 ----
        self._step_uav_decisions.clear()
        self._step_uav_delays.clear()
        self._step_uav_energies.clear()
        self._step_uav_task_volumes.clear()
        self.HAP.reset_relay_state()
        self._generate_and_cluster_tasks()
        self.total_steps += 1

        return rewards

    # ==================== 查询接口 ====================
    def get_uav_count(self) -> int:
        return self.num_uavs

    def get_available_cluster_count(self) -> int:
        return len(self.HAP.cluster_queue)

    def get_total_queue_gflops(self) -> float:
        return self.Q

    def get_lyapunov_state(self) -> Tuple[float, float, float]:
        return self.Q, self.Z, self.Y

    def get_step_stats(self) -> Dict:
        """返回当前时隙的统计信息"""
        return {
            'Q': self.Q, 'Z': self.Z, 'Y': self.Y,
            'n_clusters': len(self.HAP.cluster_queue),
            'n_remaining': len(self.HAP.remaining_task_queue) if isinstance(self.HAP.remaining_task_queue, (list, np.ndarray)) else 0,
            'total_steps': self.total_steps,
        }

    # ==================== 渲染 ====================
    def render(self):
        """渲染当前场景。

        Returns:
            render_mode="rgb_array" 时返回 (H, W, 3) uint8 数组
        """
        if self.render_mode != "rgb_array":
            raise NotImplementedError(
                f"仅支持 rgb_array 渲染, 当前 render_mode={self.render_mode!r}")

        fig = matplotlib.figure.Figure(figsize=(8, 8), dpi=100)
        canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        # 区域边框
        ax.set_xlim(0, AREA_SIZE)
        ax.set_ylim(0, AREA_SIZE)
        ax.set_aspect('equal')
        ax.set_title(f'UAV-HAP MEC Offloading  (slot {self.total_steps})')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        # 波束覆盖圈 + 当前簇负载标注
        cluster_load = {}
        for cluster in self.HAP.cluster_queue:
            key = (round(float(cluster['center_x'])), round(float(cluster['center_y'])))
            cluster_load[key] = float(cluster['total_gflops'])
        for bx, by in BEAM_POSITIONS:
            circ = matplotlib.patches.Circle((bx, by), BEAM_RADIUS,
                                             fill=True, alpha=0.08,
                                             edgecolor='tab:blue', linewidth=1.5)
            ax.add_patch(circ)
            load = cluster_load.get((bx, by))
            label = f'{load:.0f} GF' if load else 'idle'
            ax.annotate(label, (bx, by), ha='center', va='center',
                        fontsize=8, color='tab:blue')

        # 波束外零散任务散点
        if len(self.HAP.remaining_task_queue) > 0:
            arr = np.array(self.HAP.remaining_task_queue, dtype=TASK_DTYPE)
            ax.scatter(arr['x'], arr['y'], s=6, c='gray', alpha=0.5, label='scattered tasks')

        # UAV: 颜色区分算力档, 空心 = 忙碌
        tier_colors = {F_MIN: 'tab:red', F_MID: 'tab:orange', F_MAX: 'tab:green'}
        tier_names = {F_MIN: 'low', F_MID: 'mid', F_MAX: 'high'}
        seen_tiers = set()
        for uav in self.uavs:
            color = tier_colors.get(uav.Power_c, 'tab:purple')
            label = f'{tier_names.get(uav.Power_c, "?")}-tier UAV' if uav.Power_c not in seen_tiers else None
            seen_tiers.add(uav.Power_c)
            face = 'none' if uav.state != 0 else color
            ax.scatter(uav.pos[0], uav.pos[1], marker='^', s=90,
                       edgecolors=color, facecolors=face, linewidths=1.8,
                       label=label)

        # 信息框: Lyapunov 队列状态
        info = (f'Q = {self.Q:.0f} GFLOPS\n'
                f'Z = {self.Z:.0f} (energy debt)\n'
                f'Y = {self.Y:.1f} (rate debt)')
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

        ax.legend(loc='upper right', fontsize=8)
        fig.tight_layout()
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
