"""hap.py — 高空平台 (HAP) 与固定波束任务分配

HAP 悬停于区域中心上方, 维护全局任务队列:
- FixedBeamAssigner 将波束半径内的任务按最近距离聚成"波束簇";
- 波束外的零散任务留在 remaining_task_queue, 供 UAV 本地收集或中继打包;
- resolve_relay_mode 在中继 UAV 全部注册后统一结算双跳 (地面->HAP->UAV) 时延与能耗。
"""

import math
from typing import List, Tuple, Dict

import numpy as np

from config import *
from channel_task import *

# 簇/波束的数据类型
CLUSTER_DTYPE = np.dtype([
    ('center_x', 'f4'),
    ('center_y', 'f4'),
    ('total_gflops', 'f8'),
    ('task_ids', 'object')
])


class FixedBeamAssigner:
    """固定波束任务分配器

    任务按距离直接分配到最近的固定波束 (在波束半径内, 且加入后簇总量
    不超过 max_gflops); 波束外的任务进入零散队列。
    由于大部分任务已按高斯热点集中在波束周围, 无需再跑 DBSCAN 预聚类。
    """

    def __init__(self, beam_positions=None, beam_radius=95.0, max_gflops=63.0):
        self.beam_positions = beam_positions or BEAM_POSITIONS
        self.beam_radius = beam_radius
        self.max_gflops = max_gflops
        self.num_beams = len(self.beam_positions)

    def assign(self, tasks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将任务分配给固定波束 -> (clusters, remaining)"""
        n = len(tasks)
        if n == 0:
            return (np.empty(0, dtype=CLUSTER_DTYPE), np.empty(0, dtype=TASK_DTYPE))

        beam_tasks = {i: [] for i in range(self.num_beams)}
        beam_gflops = {i: 0.0 for i in range(self.num_beams)}
        remaining_indices = []

        for i in range(n):
            tx, ty = tasks[i]['x'], tasks[i]['y']
            gflops = tasks[i]['total_gflops']

            best_beam, best_dist = None, float('inf')
            for b in range(self.num_beams):
                bx, by = self.beam_positions[b]
                dist = math.hypot(tx - bx, ty - by)
                if dist <= self.beam_radius and dist < best_dist:
                    if beam_gflops[b] + gflops <= self.max_gflops:
                        best_beam, best_dist = b, dist

            if best_beam is not None:
                beam_tasks[best_beam].append(i)
                beam_gflops[best_beam] += gflops
            else:
                remaining_indices.append(i)

        clusters_list = []
        for b in range(self.num_beams):
            indices = beam_tasks[b]
            if len(indices) == 0:
                continue
            bx, by = self.beam_positions[b]
            clusters_list.append({
                'center_x': float(bx), 'center_y': float(by),
                'total_gflops': beam_gflops[b],
                'task_ids': [int(tasks[i]['id']) for i in indices]
            })

        if clusters_list:
            clusters_array = np.array(
                [(c['center_x'], c['center_y'], c['total_gflops'], c['task_ids'])
                 for c in clusters_list], dtype=CLUSTER_DTYPE)
        else:
            clusters_array = np.empty(0, dtype=CLUSTER_DTYPE)

        remaining_array = np.array([tasks[i] for i in remaining_indices], dtype=TASK_DTYPE) \
            if remaining_indices else np.empty(0, dtype=TASK_DTYPE)

        return clusters_array, remaining_array


class HAP:
    def __init__(self, center_x, center_y):
        self.center        = np.array([center_x, center_y])
        self.high          = H_hap  # 平台高度
        self.radius        = 0  # 覆盖半径, 预留
        self.Hm            = 0  # 信道增益, 预留
        self.ym            = 0  # 瞬时信噪比, 预留
        self.Power_c       = 0  # 算力, 预留
        self.quality       = 0  # 波束质量, 预留
        self.date_rate     = 0  # 数据传输速率, 预留 (每个无人机距离不同, 数值不同)

        self.state         = 0  # 0-空闲, 1-忙碌: 作为中继接收地面任务并转发

        self.UAV_num       = 0  # 无人机数量
        self.relay_UAV_num = 0  # 当前选择中继模式的无人机数量
        self.cluster_num   = 0  # 当前簇的数量
        self.cluster_max   = 0  # 簇的最大数量

        self.task_queue_total   = 0  # 总的任务队列长度
        self.task_queue_current = 0  # 当前任务队列长度
        self.task_total    = 0  # 任务总量
        self.task_total_current = 0  # 当前任务总量

        self.connected_drones = []  # 当前选择中继模式的无人机 ID
        self.task_queue       = []  # 任务队列 (元组列表, 字段同 TASK_DTYPE)
        self.cluster_queue    = []  # 已聚类且还没被选择的簇
        self.clustered_tasks_array = []  # 已被包含在簇中的任务
        self.remaining_task_queue = []  # 聚类后剩余的未聚类零散任务
        self.taskpackage_queue  = np.empty(0, dtype=TASK_DTYPE)  # 本时隙中继模式待转发的任务

        self.beam_assigner = FixedBeamAssigner(
            beam_positions=BEAM_POSITIONS, beam_radius=BEAM_RADIUS, max_gflops=W_MAX
        )

    # ==================== 波束聚类 ====================
    def HAP_clustering(self):
        """对当前任务队列执行固定波束聚类, 更新 cluster_queue / remaining_task_queue"""
        if isinstance(self.task_queue, list):
            tasks_array = np.array(self.task_queue, dtype=TASK_DTYPE)
        else:
            tasks_array = self.task_queue

        clusters, remaining = self.beam_assigner.assign(tasks_array)

        self.cluster_queue = clusters         # 数组, 字段: center_x, center_y, total_gflops, task_ids
        self.remaining_task_queue    = remaining        # 数组, 字段同 TASK_DTYPE

        # 生成一个数组, 包含所有被聚类到簇中的任务
        if len(clusters) > 0:
            id_to_idx = {task['id']: idx for idx, task in enumerate(tasks_array)}
            clustered_tasks = []
            for cluster in clusters:
                for task_id in cluster['task_ids']:
                    if task_id in id_to_idx:
                        clustered_tasks.append(tasks_array[id_to_idx[task_id]])
            self.clustered_tasks_array = np.array(clustered_tasks, dtype=TASK_DTYPE)
        else:
            self.clustered_tasks_array = np.empty(0, dtype=TASK_DTYPE)

    # ==================== 队列维护 ====================
    def add_tasks_to_hap_queue(self, tasks):
        """将生成的任务 (结构化数组) 追加到 HAP 任务队列"""
        if len(tasks) == 0:
            return
        for task in tasks:
            self.task_queue.append((
                task['id'], task['x'], task['y'],
                task['data_bits'], task['cycles_per_bit'],
                task['arrival_time'], task['total_gflops']
            ))

    def _remove_tasks_by_ids(self, task_ids):
        """从任务队列中批量删除指定 ID 的任务, 返回删除数量"""
        removed_count = 0
        # 从后向前遍历, 避免索引错位
        for i in range(len(self.task_queue) - 1, -1, -1):
            if self.task_queue[i][0] in task_ids:
                del self.task_queue[i]
                removed_count += 1
        return removed_count

    def _remove_clusters_by_indices(self, indices):
        """从 cluster_queue 中删除指定索引的簇 (单簇或多簇合并后调用)"""
        if not indices:
            return
        cluster_list = list(self.cluster_queue)
        # 倒序删除索引, 避免索引错位
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(cluster_list):
                del cluster_list[idx]
        if cluster_list:
            self.cluster_queue = np.array(cluster_list, dtype=CLUSTER_DTYPE)
        else:
            self.cluster_queue = np.empty(0, dtype=CLUSTER_DTYPE)

    def task_package_by_size(self, target_gflops, max_gflops=None):
        """从零散任务队列开头提取任务, 组成接近 target_gflops 的任务包。

        参数:
            target_gflops: 目标任务包大小 (GFLOPS)
            max_gflops: 最大允许的任务包大小, 超过则停止添加下一个任务

        返回:
            task_array: 结构化数组 (dtype=TASK_DTYPE), 包含提取的任务
            remaining_queue: 剩余零散任务列表
        """
        if len(self.remaining_task_queue) == 0:
            return np.empty(0, dtype=TASK_DTYPE), []

        total = 0.0
        collected = []
        extract_until = 0

        for i, task in enumerate(self.remaining_task_queue):
            gflops = task[6]  # total_gflops
            if total + gflops <= target_gflops:
                total += gflops
                collected.append(task)
                extract_until = i + 1
            elif max_gflops is not None and total + gflops <= max_gflops:
                total += gflops
                collected.append(task)
                extract_until = i + 1
                break
            else:
                break

        remaining_queue = self.remaining_task_queue[extract_until:]

        if collected:
            task_array = np.array(collected, dtype=TASK_DTYPE)
        else:
            task_array = np.empty(0, dtype=TASK_DTYPE)

        return task_array, remaining_queue

    def collect_task_package_in_coverage(self, x, y, coverage_radius: float) -> np.ndarray:
        """收集 UAV 覆盖半径内的所有零散任务, 合并为任务包。

        参数:
            x, y: 无人机的水平坐标 (米)
            coverage_radius: 覆盖半径 (米) - 水平距离阈值

        返回:
            (结构化数组, 剩余零散任务列表); 无任务时返回空数组
        """
        uav_x, uav_y = x, y
        collected_tasks = []
        remaining_queue = []

        for task in self.remaining_task_queue:        # 从不在簇中的任务里选择
            tx, ty = task[1], task[2]
            dist = math.hypot(uav_x - tx, uav_y - ty)
            if dist <= coverage_radius:
                collected_tasks.append(task)
            else:
                remaining_queue.append(task)

        if collected_tasks:
            return np.array(collected_tasks, dtype=TASK_DTYPE), remaining_queue
        else:
            return np.empty(0, dtype=TASK_DTYPE), remaining_queue

    # ==================== 中继模式统一结算 ====================
    def reset_relay_state(self):
        """每个时隙开始时重置中继状态"""
        self.relay_UAV_num = 0
        self.connected_drones = []
        self.taskpackage_queue = np.empty(0, dtype=TASK_DTYPE)

    def resolve_relay_mode(self, uavs_dict, task_array: np.ndarray,
                           bandwidth: float = B, freq_hz: float = F_c):
        """在中继 UAV 全部注册后统一结算时延与能耗。

        中继链路: 地面设备 -> HAP 上行 (所有中继 UAV 共享) -> HAP 下行 (带宽均分) -> UAV 计算。

        参数:
            uavs_dict:  {uav_id: UAV对象} 字典, 包含所有无人机
            task_array: 当前 HAP 任务队列数组 (TASK_DTYPE) (保留参数, 兼容外部调用)
            bandwidth:  系统总带宽 (Hz)
            freq_hz:    载波频率 (Hz)

        返回:
            results: {uav_id: {'delay': float, 'energy': float, 'task_data_volume': float}}
                     只包含选择了中继模式的 UAV
        """
        results = {}
        relay_count = len(self.connected_drones)
        if relay_count == 0 or len(self.taskpackage_queue) == 0:
            return results

        # ---- 1. 地面->HAP 上行总传输时延 ----
        total_g2h_time = 0.0
        total_gflops_relay = 0.0
        for task in self.taskpackage_queue:
            dx = task['x'] - self.center[0]
            dy = task['y'] - self.center[1]
            dist_g2h = math.sqrt(dx * dx + dy * dy + H_hap ** 2)
            rate_g2h = R_transmission_rate_1(dist_g2h, G_tx_ground, G_rx_hap, P_tx_ground, bandwidth)
            total_g2h_time += task['data_bits'] / rate_g2h
            total_gflops_relay += task['total_gflops']

        # ---- 2. HAP->UAV 下行传输 + 计算 ----
        for uav_id in self.connected_drones:
            uav = uavs_dict[uav_id]
            dx = uav.pos[0] - self.center[0]
            dy = uav.pos[1] - self.center[1]
            dz = uav.pos[2] - H_hap
            dist_h2u = math.sqrt(dx * dx + dy * dy + dz * dz)

            # 下行传输速率 (带宽均分给所有中继 UAV)
            rate_h2u = R_transmission_rate_2(dist_h2u, G_tx_hap, G_rx_uav, P_tx_hap, bandwidth, relay_count)

            total_data_bits = sum(task['data_bits'] for task in self.taskpackage_queue)
            tx_time_h2u = total_data_bits / rate_h2u

            comp_time = total_gflops_relay / uav.Power_c  # GFLOPS / Hz = 秒
            # G->H 上行时间由所有中继 UAV 共享, 计入每个 UAV 的总延迟
            total_delay = total_g2h_time + tx_time_h2u + comp_time

            # 能耗: 计算能耗 (通信能耗暂忽略)
            comp_energy = uav.calculate_power(total_gflops_relay * 1e9)  # GFLOPS -> FLOPS

            results[uav_id] = {
                'delay': total_delay,
                'energy': comp_energy,
                'task_data_volume': total_gflops_relay
            }

        # ---- 3. 清理: 从任务队列移除已处理的中继任务 ----
        relay_task_ids = set(task['id'] for task in self.taskpackage_queue)
        self._remove_tasks_by_ids(list(relay_task_ids))
        self.taskpackage_queue = np.empty(0, dtype=TASK_DTYPE)

        return results
