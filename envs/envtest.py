from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle

import numpy as np
from gym import spaces
from gym.utils import seeding
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env import utils


class RSU:
    def __init__(self, position, vertical_distance, coverage_radius):
        """
        初始化RSU类。
        :param position: RSU在道路上的位置 (x, y) 坐标
        :param vertical_distance: RSU与道路之间的垂直距离
        :param coverage_radius: RSU的传输覆盖半径
        """
        self.position = position
        self.vertical_distance = vertical_distance  # RSU与道路之间的垂直距离
        self.coverage_radius = coverage_radius  # RSU的传输覆盖半径
        self.state = None

    def receive_state(self, vehicle_state):
        """
        接收来自车辆的状态。
        :param vehicle_state: 车辆的状态，包括位置和速度等
        """
        self.state = vehicle_state

    def calculate_distance(self, vehicle_position):
        """
        计算车辆与RSU之间的距离。
        :param vehicle_position: 车辆的位置 (x, y) 坐标
        :return: 距离值
        """
        rsu_x, rsu_y = self.position
        vehicle_x, vehicle_y = vehicle_position
        return np.sqrt((rsu_x - vehicle_x) ** 2 + (rsu_y - vehicle_y) ** 2 + self.vertical_distance ** 2)

    def calculate_channel_gain(self, vehicle_position, power_transmit, interference_power, noise_density,
                               bandwidth, fc=5.9):
        """
        计算信道增益。
        :param vehicle_position: 车辆的位置 (x, y) 坐标
        :param power_transmit: 车辆发射功率
        :param interference_power: 干扰功率
        :param noise_density: 噪声功率密度
        :param bandwidth: 通信带宽
        :param fc: 载波频率，默认为 5.9 GHz
        :return: 信道容量（单位：比特每秒）
        """
        # 计算车辆和RSU之间的距离
        distance = self.calculate_distance(vehicle_position)

        # 计算路径损耗 PL (单位为 dB)
        path_loss = 28 + 22 * np.log10(distance) + 20 * np.log10(fc)

        # 计算信道增益 h(t)
        channel_gain = 10 ** (-path_loss / 10)

        # 计算信道增益的功率
        effective_power = power_transmit * channel_gain

        # 计算信道容量 R(t)
        sinr = effective_power / (noise_density * bandwidth + interference_power)
        channel_capacity = bandwidth * np.log2(1 + sinr)

        return channel_capacity

    def calculate_transmission_delay(self, data_size, channel_capacity):
        """
        计算数据从车辆传输到RSU的时延。
        :param data_size: 数据大小（单位：比特）
        :param channel_capacity: 信道容量（单位：比特每秒）
        :return: 传输时延（单位：秒）
        """
        if channel_capacity == 0:
            return float('inf')  # 如果信道容量为 0，时延为无穷大（传输不可能完成）
        return data_size / channel_capacity

    def clear(self):
        self.state = None


class testEnv(AbstractEnv):
    def __init__(self, render_mode=None, config=None):
        self._initialized = False
        super().__init__(config=config)

        # 计算 RSU 的位置
        rsu_x = self.config["distance"] / 2  # 道路中心
        lane_width = 3.5  # 单车道宽度
        rsu_y = (self.config["lanes_count"] * lane_width) / 2  # 道路宽度的中心
        rsu_position = (rsu_x, rsu_y)  # 中心位置

        vertical_distance = 25  # RSU 与道路的垂直距离
        coverage_radius = 5000  # RSU 的覆盖半径

        # 初始化 RSU
        self.rsu = RSU(position=rsu_position, vertical_distance=vertical_distance,
                       coverage_radius=coverage_radius)
        self.controlled_vehicle = None
        self.render_mode = render_mode

        # 在所有属性初始化完成后再调用 reset()
        self._initialized = True
        self.reset()

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 14,  # 仿真频率
                "lanes_count": 2,  # 车道数量
                "vehicles_count": 25,  # 车辆数量
                "distance": 2000,  # 规定行驶距离 [m]
                "ego_spacing": 2,  # 自车间距
                "controlled_vehicles": 1,  # 受控车辆数量，添加默认值
                "observation": {"type": "CustomObservation"},  # 观察类型
                "action": {"type": "DiscreteMetaAction"},  # 动作类型
                "initial_lane_id": None,  # 初始车道ID
                "vehicles_density": 0.8,  # 车辆密度
                "lane_change_reward": 10,  # 每次变道的基础奖励，动态调整
                "collision_penalty": -15,  # 增加碰撞惩罚
                "high_speed_reward": 10,  # 提高高速奖励
                "delay_reward_factor": 10,  # 加强传输时延奖励
                "reward_speed_range": [60, 100],  # 奖励速度范围
                "normalize_reward": True,  # 是否归一化奖励
                "offroad_terminal": False,  # 是否允许越界终止
            }
        )
        return cfg

    def step(self, action):
        # 调用父类的 step 方法
        result = super().step(action)
        obs, reward, done = result[:3]
        info = result[3] if len(result) > 3 and isinstance(result[3], dict) else {}
        truncated = info.get('TimeLimit.truncated', False)

        # 确保受控车辆已经初始化
        if self.controlled_vehicle is None:
            raise RuntimeError("Controlled vehicle is not set.")

        # 计算车辆与 RSU 的信道增益
        vehicle_position = self.controlled_vehicle.position[:2]
        power_transmit = 1.0  # 假设车辆的发射功率为 1 W
        interference_power = 0  # 假设其他干扰功率为0 无干扰
        noise_density = 1e-9  # 假设噪声密度为 1e-9 W/Hz
        bandwidth = 1e6  # 假设带宽为 1 MHz

        # 通过 RSU 计算信道容量
        channel_capacity = self.rsu.calculate_channel_gain(
            vehicle_position=vehicle_position,
            power_transmit=power_transmit,
            interference_power=interference_power,
            noise_density=noise_density,
            bandwidth=bandwidth
        )

        # 计算传输的数据大小
        data_size = self.calculate_data_size()

        # 计算传输时延
        transmission_delay = self.rsu.calculate_transmission_delay(data_size=data_size,
                                                                   channel_capacity=channel_capacity)

        # 将信道容量和传输时延添加到 info 字典中
        info['channel_capacity'] = channel_capacity
        info['transmission_delay'] = transmission_delay

        # RSU 接收车辆状态
        vehicle_state = self.vehicle_state()
        self.rsu.receive_state(vehicle_state)

        # 自定义奖励逻辑
        reward = self._reward(action, transmission_delay)

        return obs, reward, done, truncated, info

    def calculate_data_size(self):
        """
        计算车辆状态数据的大小（单位：比特）。
        :return: 数据大小（单位：比特）
        """
        # 获取车辆的观测状态
        vehicle_state = self.vehicle_state()

        # 数据大小计算（假设每个浮点数占用32位，即4字节）
        # 每个状态量为浮点数，因此数据大小为状态量数量乘以4字节，再乘以8得到比特数
        state_size = len(vehicle_state) * 4 * 8  # 以比特为单位

        # 如果包含图像信息，需要额外计算图像数据的大小
        image_data_size = 0
        if self.render_mode == 'rgb_array':
            image = self.observation_type.observe_image()  # 获取图像观测
            if image is not None:
                image_data_size = image.size * 8  # 假设图像为8位（1字节）每像素

        # 总数据大小
        total_data_size = state_size + image_data_size

        return total_data_size

    def vehicle_state(self):
        # 使用 controlled_vehicle 获取状态
        if self.controlled_vehicle is None:
            raise RuntimeError("controlled_vehicle not set.")
        return np.array([
            self.controlled_vehicle.position[0],
            self.controlled_vehicle.position[1],
            self.controlled_vehicle.speed,
            float(self.controlled_vehicle.crashed)  # 添加观测终止标志（是否发生碰撞）
        ])

    def reset(self, seed=None, **kwargs):
        if not self._initialized:
            return
        self.np_random, seed = seeding.np_random(seed)
        self._reset()
        self.rsu.clear()
        if not hasattr(self, 'controlled_vehicles') or not self.controlled_vehicles:
            raise RuntimeError("controlled_vehicles has not been properly initialized.")
        self.controlled_vehicle = self.controlled_vehicles[0]
        observation = self.observation_type.observe()
        return observation, {}

    def _reset(self) -> None:
        """重置环境状态，创建道路和车辆。"""
        self._create_road()  # 调用方法创建道路
        self._create_vehicles()  # 调用方法创建车辆

    def _create_road(self) -> None:
        """创建一条由直线相邻车道组成的道路。"""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30  # 创建直线道路网络
            ),
            np_random=self.np_random,
            record_history=self.config.get("show_trajectories", False),  # 是否记录历史轨迹
        )

    def _create_vehicles(self) -> None:
        """创建新的随机车辆，并将其添加到道路上。"""
        other_vehicles_type = utils.class_from_path(
            self.config.get("other_vehicles_type", "highway_env.vehicle.kinematics.Vehicle"))  # 获取其他车辆类型
        other_per_controlled = utils.near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]  # 分配车辆数量
        )

        self.controlled_vehicles = []  # 初始化受控车辆列表

        # 创建受控车辆
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=50,  # 初始速度提高到 50
                lane_id=self.config["initial_lane_id"],  # 初始车道ID
                spacing=self.config["ego_spacing"],  # 自车间距
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed  # 创建受控车辆
            )
            self.controlled_vehicles.append(vehicle)  # 添加到受控车辆列表
            self.road.vehicles.append(vehicle)  # 添加到道路上的车辆列表

            # 创建其他随机车辆
            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]  # 创建随机车辆
                )

                # 设置恒定速度和禁用换道及其他行为
                vehicle.speed = 20  # 设置恒定速度为 20
                vehicle.target_speed = 20  # 确保目标速度也是 20

                # 禁用换道计划方法
                vehicle.plan_route_to = lambda destination: None
                # 禁用随机化行为
                vehicle.randomize_behavior = lambda: None
                # 禁用动作控制，包括加速、减速和换道
                vehicle.act = lambda action=None: None  # 修改 lambda 函数，接受可选的 `action` 参数

                # 禁用其他的加速逻辑
                vehicle.controller = None  # 去掉控制器，以防止自动控制调整速度
                vehicle.check_collisions = False  # 禁用非受控车辆的碰撞检查

                # 添加到道路上的车辆列表
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action, transmission_delay: float = 0.0) -> float:
        rewards = self._rewards(action)
        safety_distance = 10  # 安全距离设定
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        distance_to_front = (
            front_vehicle.position[0] - self.controlled_vehicle.position[0]
            if front_vehicle
            else float('inf')
        )

        # 调整奖励权重
        reward = (
                1.5 * rewards["collision_reward"] +  # 增加碰撞惩罚权重
                1.0 * rewards["lane_change_reward"] +  # 调整换道奖励权重
                0.8 * rewards["high_speed_reward"]  # 增加高速奖励权重
        )

        # 鼓励安全换道
        if action in [0, 2] and distance_to_front > safety_distance:
            reward += 5

        # 鼓励加速行为
        if action == 3 and (front_vehicle is None or distance_to_front > safety_distance):
            reward += 3

        # 传输时延奖励
        delay_reward = max(0.0, (5.0 - transmission_delay))*10
        reward += delay_reward * self.config["delay_reward_factor"]

        # 动态范围裁剪
        reward = np.clip(reward, -50, 50)

        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        """
        计算各类奖励的具体值。

        参数:
        - action (Action): 当前采取的动作。

        返回值:
        - dict[str, float]: 奖励名称及其对应的值。
        """
        forward_speed = self.controlled_vehicle.speed  # 当前车辆速度
        safety_distance = 20  # 安全距离
        front_vehicle, _ = self.road.neighbour_vehicles(self.controlled_vehicle)
        distance_to_front = (
            front_vehicle.position[0] - self.controlled_vehicle.position[0]
            if front_vehicle
            else float('inf')
        )

        # 速度奖励归一化
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )

        # 奖励定义
        if self.controlled_vehicle.crashed and (front_vehicle is not None and front_vehicle.crashed):
            collision_reward = -10
        elif self.controlled_vehicle.crashed:
            collision_reward = -5
        else:
            collision_reward = 0.0
        lane_change_reward = (
            5  # 基础奖励
            if action in [0, 2] and distance_to_front > safety_distance  # 换道且距离足够安全
            else 0.0
        )
        high_speed_reward = np.clip(scaled_speed, 0, 1) * 8
        # 增加距离前车过近的惩罚
        proximity_penalty = -5 if distance_to_front < safety_distance else 0.0

        # 增加长时间静止的惩罚
        stationary_penalty = -3 if self.controlled_vehicle.speed < 22 else 0.0

        # 返回奖励字典
        return {
            "collision_reward": collision_reward,
            "lane_change_reward": lane_change_reward,
            "high_speed_reward": high_speed_reward,
            "proximity_penalty": proximity_penalty,
            "stationary_penalty": stationary_penalty,
        }

    def _is_terminated(self) -> bool:
        return (
                self.controlled_vehicle.crashed or  # 碰撞
                self.controlled_vehicle.position[0] >= self.config["distance"] or  # 达到规定行驶距离
                (self.config["offroad_terminal"] and not self.controlled_vehicle.on_road)  # 偏离道路
        )

    def _is_truncated(self) -> bool:
        """如果达到规定行驶距离则事件被截断。"""
        return self.controlled_vehicle.position[0] >= self.config["distance"]  # 达到规定行驶距离

    def render(self, mode='human'):
        """
        渲染环境。
        :param mode: 渲染模式（可选的，如 'human', 'rgb_array' 等）
        """
        # 如果父类的 render 方法存在且不接受 mode 参数
        if hasattr(super(), 'render'):
            try:
                # 尝试调用父类的 render 方法，看看是否接受 mode 参数
                super().render(mode=mode)
            except TypeError:
                # 如果不接受 mode 参数，就调用不带参数的版本
                super().render()
        else:
            # 如果父类没有 render 方法或需要自定义的渲染逻辑
            if mode == 'human':
                # 如果需要可视化
                print("Rendering in human mode... (this is a placeholder)")
            elif mode == 'rgb_array':
                # 如果需要返回图像
                print("Rendering as RGB array... (this is a placeholder)")
            else:
                raise ValueError(f"Unsupported render mode: {mode}")
