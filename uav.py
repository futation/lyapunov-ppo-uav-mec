"""uav.py — 无人机实体

模拟无人机的传输/计算时延、能耗、飞行功率，以及多时隙忙碌状态机：
一次决策（飞行 + 传输 + 计算）可能跨越多个时隙，由 tick() 逐时隙推进，
任务数据暂存在 pending 字段中，完成时由环境结算。
"""

import math
from typing import List, Dict

import numpy as np

from config import *
from channel_task import *


class UAV:
    def __init__(self, UAV_id: int):
        """初始化无人机实例

        Args:
            UAV_id: 无人机唯一标识符
        """
        self.id          = UAV_id
        self.pos         = [0.0, 0.0, 300.0]  # 三维坐标
        self.initial_pos = [0.0, 0.0, 300.0]  # 用于重置每回合初始坐标
        self.state       = 0            # (0-空闲, 1-忙碌)
        self.decision    = 0            # (0-无, 1-本地, 2-集群, 3-中继)
        self.Power_c     = 0.0          # 计算频率 (Hz)
        self.task        = 0.0          # 目前接收的任务大小
        self.high        = 0.0          # 高度
        self.cluster     = []           # 选择的簇
        self.cluster_num = 0            # 选择的簇的数量
        self.speed       = float(UAV_SPEED)  # 水平速度

        self.p_fly       = 0.0          # 飞行功率

        self.transmission_rate = 0.0    # 传输速率
        self.path_loss   = 0.0          # 路径损耗
        self.gain        = 0.0          # 增益
        self.SNR         = 0.0          # 信噪比
        self.R_relay     = 0.0          # 中继模式下行传输速率

        self.delay       = 0.0
        self.energy      = 0.0
        self.utility     = 0.0          # 效用函数

        self.current_time = 0.0         # 用于记录时间步
        self.need_time   = 0.0          # 一次决策所需时间
        self.need_energy = 0.0          # 一次决策所需能量

        self.task_data_volume = 0.0     # 本次决策接受的数据总量 (GFLOPS)
        self._fly_dist       = 0.0      # 飞行距离 (用于奖励距离惩罚)

        # ---- 多时隙忙碌追踪 ----
        self.busy_remaining = 0         # 剩余忙碌时隙数 (0=空闲)
        self.target_x        = 0.0      # 飞行目标 X
        self.target_y        = 0.0      # 飞行目标 Y
        self.is_flying       = False    # 是否在飞行阶段 (True=飞行, False=计算)
        self._flight_slots   = 0        # 飞行阶段需要的总时隙数
        self._compute_slots  = 0        # 计算阶段需要的总时隙数
        self._pending_task_vol = 0.0    # 待完成的任务量 (GFLOPS, 完成时结算奖励)
        self._pending_delay    = 0.0    # 待结算的总延迟 (秒)
        self._pending_energy   = 0.0    # 待结算的总能耗 (J)
        self._action_taken     = 0      # 记录采取的动作用于奖励计算

    # ==================== 时延: 本地计算 ====================
    def local_transmission(self, task):
        """地面设备 -> UAV 上行传输 + UAV 本地计算的总结时延 (秒)"""
        distance = calculate_distance(self.pos[0], task['x'], self.pos[1], task['y'], self.pos[2], 0)
        R_G_E    = R_transmission_rate_1(distance, G_tx_ground, G_rx_uav, P_tx_ground, B_UAV)
        time_G_E = task['data_bits'] / R_G_E
        time_calculate = task['total_gflops'] / self.Power_c
        time     = time_G_E + time_calculate
        return time

    # ==================== 时延: 飞向簇 ====================
    def fly_time(self, cluster_center):
        D    = math.sqrt((self.pos[0] - cluster_center[0])**2 + (self.pos[1] - cluster_center[1])**2)
        time = D / self.speed
        return time

    def Select_single_cluster(self, cluster, task_array):
        """计算 UAV 飞向单簇并处理所有任务的总时延 (不更新位置, 由 tick() 逐步移动)

        :return: total_time, flight_time
        """
        task_id_to_idx = {task['id']: idx for idx, task in enumerate(task_array)}

        # 飞行时间
        flight_time = self.fly_time((cluster['center_x'], cluster['center_y']))
        total_time = flight_time

        # 所有任务的传输+计算时间
        for task_id in cluster['task_ids']:
            idx = task_id_to_idx[task_id]
            task = task_array[idx]
            total_time += self.local_transmission(task)

        return total_time, flight_time

    # ==================== 时延: 多簇合并 ====================
    def check_multi_cluster_feasibility(self, clusters):
        """检查多簇组合是否可行

        :param clusters: 簇字典列表, 每个字典包含 'center_x', 'center_y'
        :return: bool
        """
        # 1. 算力条件: 最多可选簇数 = floor(算力 / F_MIN)
        max_clusters = int(self.Power_c // F_MIN)   # Power_c 单位 Hz
        if len(clusters) > max_clusters:
            return False
        # 2. 距离条件: 所有簇中心两两距离不超过 max_distance
        centers = [(c['center_x'], c['center_y']) for c in clusters]
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.hypot(dx, dy)
                if dist > 100.0:
                    return False
        return True

    def compute_weighted_center(self, clusters):
        """计算多簇的加权重心 (按每个簇的 total_gflops 加权)

        :return: (center_x, center_y, total_gflops)
        """
        total_w = sum(c['total_gflops'] for c in clusters)
        if total_w == 0:
            return 0.0, 0.0, 0.0
        sum_x = sum(c['total_gflops'] * c['center_x'] for c in clusters)
        sum_y = sum(c['total_gflops'] * c['center_y'] for c in clusters)
        return sum_x / total_w, sum_y / total_w, total_w

    def Select_multi_cluster(self, clusters, task_array):
        """计算 UAV 飞向多簇 (合并簇) 并处理所有任务的总时延

        :param clusters: 簇字典列表, 每个字典包含 'center_x', 'center_y', 'task_ids', 'total_gflops'
        :param task_array: 任务结构化数组 (TASK_DTYPE), 包含所有当前存在的任务
        :return: (总时延秒, 飞行时间秒), 不可行返回 (inf, 0)
        """
        # 可行性检查
        if not self.check_multi_cluster_feasibility(clusters):
            return float('inf'), 0.0

        # 计算加权重心
        cx, cy, total_gflops = self.compute_weighted_center(clusters)

        # 飞行时间 (从当前位置到重心; 不在此处修改 pos,
        # 实际位移由 start_beam_task 记录起点后由 tick() 逐步插值)
        flight_time = self.fly_time((cx, cy))

        # 构建任务ID -> 数组索引的映射
        task_id_to_idx = {task['id']: idx for idx, task in enumerate(task_array)}

        # 累加所有任务的传输和计算时间
        total_comp_time = 0.0
        for cluster in clusters:
            for task_id in cluster['task_ids']:
                idx = task_id_to_idx[task_id]
                task = task_array[idx]
                total_comp_time += self.local_transmission(task)

        total_time = flight_time + total_comp_time
        return total_time, flight_time

    def get_feasible_cluster_combinations(self, clusters_array: np.ndarray, max_distance: float = 100.0) -> List[Dict]:
        """根据无人机算力和簇间距离, 生成所有可行的单簇和多簇组合。

        参数:
            clusters_array: 结构化数组, dtype=CLUSTER_DTYPE,
                包含字段 'center_x', 'center_y', 'total_gflops', 'task_ids'
            max_distance: 多簇合并时, 簇中心之间的最大允许距离 (米)

        返回:
            列表, 每个元素为字典, 包含:
                'cluster_indices': List[int]   # 所选簇在原数组中的索引
                'total_gflops': float          # 总计算量
                'task_ids': List[int]          # 所有任务的ID列表
        """
        max_clusters = int(self.Power_c // F_MIN)   # 最大可选簇数 (基于算力)
        if max_clusters < 1:
            return []

        n_clusters = len(clusters_array)
        feasible = []

        # 1. 单簇 (所有未分配的簇都可作为单簇动作)
        for idx in range(n_clusters):
            feasible.append({
                'cluster_indices': [idx],
                'total_gflops': clusters_array[idx]['total_gflops'],
                'task_ids': clusters_array[idx]['task_ids'].copy()
            })

        # 2. 多簇组合 (仅当最大可选簇数 >= 2 时)
        if max_clusters >= 2:
            centers = [(clusters_array[i]['center_x'], clusters_array[i]['center_y']) for i in range(n_clusters)]

            def is_distance_valid(indices):
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        dx = centers[indices[i]][0] - centers[indices[j]][0]
                        dy = centers[indices[i]][1] - centers[indices[j]][1]
                        if math.hypot(dx, dy) > max_distance:
                            return False
                return True

            # 生成所有可能的组合 (2 簇、3 簇... 直到 max_clusters)
            from itertools import combinations
            for k in range(2, max_clusters + 1):
                for combo in combinations(range(n_clusters), k):
                    if is_distance_valid(combo):
                        total_gflops = sum(clusters_array[i]['total_gflops'] for i in combo)
                        task_ids = []
                        for i in combo:
                            task_ids.extend(clusters_array[i]['task_ids'])
                        feasible.append({
                            'cluster_indices': list(combo),
                            'total_gflops': total_gflops,
                            'task_ids': task_ids
                        })

        return feasible

    # ==================== 能耗 ====================
    def calculate_power(self, W_t):
        """计算能耗: 芯片电容模型 E = kappa * f^2 * 计算量"""
        E = kappa * (self.Power_c ** 2) * W_t
        return E

    def P_fly(self):
        """计算无人机飞行功率 (旋翼功率模型)

        P(V) = (M*G/V) * v_i(V) + 0.5 * rho * C_D * S * V^3
        其中 v_i(V) 为前飞诱导速度, 由悬停诱导速度 v_h 推出
        """
        v_i = math.sqrt((math.sqrt(UAV_SPEED**4 + 4 * v_h**4) - UAV_SPEED**2) / 2)
        term1 = (M * G / UAV_SPEED) * v_i      # 诱导功率
        term2 = 0.5 * RHO_AIR * C_D * S * UAV_SPEED**3   # 型阻功率
        return term1 + term2

    # ==================== 多时隙忙碌状态机 ====================

    def start_beam_task(self, cluster_center_x: float, cluster_center_y: float,
                        total_time: float, flight_time: float,
                        task_vol: float, comp_energy: float, fly_energy: float,
                        fly_dist: float, action: int):
        """开始一个波束任务 (动作 a=n), 进入忙碌状态

        Args:
            cluster_center_x/y: 波束中心坐标
            total_time: 总耗时 (飞行+传输+计算, 秒)
            flight_time: 飞行时间 (秒)
            task_vol: 任务总计算量 (GFLOPS)
            comp_energy: 计算能耗 (J)
            fly_energy: 飞行能耗 (J)
            fly_dist: 飞行距离 (m)
            action: 采取的动作编号
        """
        self.state = 1           # 忙碌
        self.decision = 2        # 波束模式
        self.target_x = cluster_center_x
        self.target_y = cluster_center_y
        self.is_flying = True
        self._fly_dist = fly_dist

        # 计算各阶段需要的时隙数
        self._flight_slots = max(1, int(np.ceil(flight_time / SLOT_DURATION)))
        compute_time = total_time - flight_time
        self._compute_slots = max(1, int(np.ceil(compute_time / SLOT_DURATION)))
        self.busy_remaining = self._flight_slots + self._compute_slots

        # 存储待结算数据
        self._pending_task_vol = task_vol
        self._pending_delay = total_time
        self._pending_energy = comp_energy + fly_energy
        self._action_taken = action

        # 记录起始位置 (用于 tick() 逐步插值移动)
        self._start_x = self.pos[0]
        self._start_y = self.pos[1]

    def start_onboard_task(self, delay: float, energy: float, task_vol: float,
                           decision: int):
        """开始机载任务 (动作 a=0: 本地=1 或 中继=3), 进入忙碌状态

        Args:
            delay: 总延迟 (传输+计算, 秒)
            energy: 总能耗 (J)
            task_vol: 任务计算量 (GFLOPS)
            decision: 1=本地计算, 3=中继模式
        """
        self.state = 1
        self.decision = decision
        self.is_flying = False
        self.target_x = self.pos[0]
        self.target_y = self.pos[1]

        self._flight_slots = 0
        self._compute_slots = max(1, int(np.ceil(delay / SLOT_DURATION)))
        self.busy_remaining = self._compute_slots

        self._pending_task_vol = task_vol
        self._pending_delay = delay
        self._pending_energy = energy
        self._action_taken = 0
        self._fly_dist = 0.0

    def tick(self) -> bool:
        """每个时隙调用一次: 推进忙碌进度 (飞行阶段线性插值移动)

        Returns:
            True  = 本时隙任务完成 (UAV 变为空闲)
            False = 仍在忙碌中
        """
        if self.busy_remaining <= 0:
            return False  # 已经空闲

        # 飞行阶段: 每时隙移动剩余路程的 1/_flight_slots
        if self.is_flying and self._flight_slots > 0:
            frac = 1.0 / max(1, self._flight_slots)
            self.pos[0] += (self.target_x - self._start_x) * frac
            self.pos[1] += (self.target_y - self._start_y) * frac
            self._flight_slots -= 1
            if self._flight_slots <= 0:
                # 飞行结束, 到达目标
                self.pos[0] = self.target_x
                self.pos[1] = self.target_y
                self.is_flying = False

        self.busy_remaining -= 1

        if self.busy_remaining <= 0:
            # 任务完成
            self.state = 0
            self.decision = 0
            self.pos[0] = self.target_x
            self.pos[1] = self.target_y
            return True

        return False

    def collect_pending_reward(self) -> tuple:
        """任务完成时调用: 返回待结算的 (task_vol, delay, energy, action, fly_dist) 并清空 pending 数据"""
        result = (self._pending_task_vol, self._pending_delay, self._pending_energy,
                  self._action_taken, self._fly_dist)
        self._pending_task_vol = 0.0
        self._pending_delay = 0.0
        self._pending_energy = 0.0
        self._action_taken = 0
        self._fly_dist = 0.0
        return result
